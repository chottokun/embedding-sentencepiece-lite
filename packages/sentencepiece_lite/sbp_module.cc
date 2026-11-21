#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include "sentencepiece_lite.h"


class FastSBPTokenizer {
public:
  FastSBPTokenizer(const std::string& model_path) {
    fd_ = open(model_path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("Could not open model file: " + model_path);
    struct stat sb;
    if (fstat(fd_, &sb) != 0) {
      close(fd_);
      throw std::runtime_error("fstat failed");
    }
    size_ = sb.st_size;
    addr_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("mmap failed");
    }
    std::string_view view(static_cast<const char*>(addr_), size_);
    processor_ = std::make_unique<sentencepiece::lite::SentencePieceLiteProcessor>(view);
    if (processor_->status() != sentencepiece::lite::StatusCode::kOk) {
      munmap(addr_, size_);
      close(fd_);
      throw std::runtime_error("SentencePieceLiteProcessor failed to load model");
    }
  }

  ~FastSBPTokenizer() {
    if (addr_ && addr_ != MAP_FAILED) munmap(addr_, size_);
    if (fd_ >= 0) close(fd_);
  }

  std::vector<int> encode(const std::string& text, size_t chunk_size = 16384, int num_threads = 0) {
    if (num_threads > 0) omp_set_num_threads(num_threads);

    std::string_view sv(text);
    std::vector<std::string_view> chunks;
    size_t start = 0;
    while (start < sv.size()) {
      if (start + chunk_size >= sv.size()) {
        chunks.push_back(sv.substr(start));
        break;
      }
      size_t split_pos = sv.find('\n', start + chunk_size);
      if (split_pos == std::string_view::npos) split_pos = sv.find(' ', start + chunk_size);
      if (split_pos != std::string_view::npos && split_pos < sv.size()) {
        chunks.push_back(sv.substr(start, (split_pos + 1) - start));
        start = split_pos + 1;
      } else {
        chunks.push_back(sv.substr(start));
        break;
      }
    }

    std::vector<std::vector<int>> per_chunk(chunks.size());
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < chunks.size(); ++i) {
      processor_->Encode(chunks[i], &per_chunk[i]);
    }

    size_t total = 0;
    for (const auto& ids : per_chunk) total += ids.size();
    std::vector<int> res;
    res.reserve(total);
    for (auto& ids : per_chunk) {
      res.insert(res.end(), ids.begin(), ids.end());
    }
    return res;
  }

private:
  int fd_ = -1;
  size_t size_ = 0;
  void* addr_ = nullptr;
  std::unique_ptr<sentencepiece::lite::SentencePieceLiteProcessor> processor_;
};

// ---------------------------------------------------------------------------
// CPython stable-ABI (limited API) binding.
// Replaces the former pybind11 module block: pybind11 (checked 2.13.6 and
// 3.1.0) does not compile under Py_LIMITED_API, so this thin glue uses only
// stable-ABI CPython functions. The Python-visible API is unchanged:
//   sentencepiece_lite.FastSBPTokenizer(model_path).encode(text, ...)
// ---------------------------------------------------------------------------
namespace {

struct TokenizerObject {
  PyObject_HEAD
  FastSBPTokenizer* impl;
};

int Tokenizer_init(PyObject* self, PyObject* args, PyObject* kwargs) {
  static const char* kwlist[] = {"model_path", nullptr};
  PyObject* path_obj = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(kwlist), &path_obj))
    return -1;
  Py_ssize_t len = 0;
  const char* path = PyUnicode_AsUTF8AndSize(path_obj, &len);
  if (path == nullptr) return -1;
  auto* obj = reinterpret_cast<TokenizerObject*>(self);
  try {
    // 再初期化時のリークを防ぐため例外安全に差し替える
    auto next = std::make_unique<FastSBPTokenizer>(std::string(path, static_cast<size_t>(len)));
    delete obj->impl;
    obj->impl = next.release();
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  return 0;
}

void Tokenizer_dealloc(PyObject* self) {
  auto* obj = reinterpret_cast<TokenizerObject*>(self);
  delete obj->impl;
  obj->impl = nullptr;
  // Heap Type の参照カウントを適切に解放する
  PyTypeObject* tp = Py_TYPE(self);
  auto free_fn = reinterpret_cast<freefunc>(PyType_GetSlot(tp, Py_tp_free));
  free_fn(self);
  Py_DECREF(tp);
}

PyObject* Tokenizer_encode(PyObject* self, PyObject* args, PyObject* kwargs) {
  static const char* kwlist[] = {"text", "chunk_size", "num_threads", nullptr};
  PyObject* text_obj = nullptr;
  Py_ssize_t chunk_size = 16384;
  int num_threads = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ni", const_cast<char**>(kwlist),
                                   &text_obj, &chunk_size, &num_threads))
    return nullptr;
  if (chunk_size <= 0) {
    PyErr_SetString(PyExc_ValueError, "chunk_size must be positive");
    return nullptr;
  }
  Py_ssize_t len = 0;
  const char* utf8 = PyUnicode_AsUTF8AndSize(text_obj, &len);
  if (utf8 == nullptr) return nullptr;
  auto* obj = reinterpret_cast<TokenizerObject*>(self);
  if (obj->impl == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Tokenizer is not initialized");
    return nullptr;
  }
  std::vector<int> ids;
  std::string error;
  bool failed = false;
  Py_BEGIN_ALLOW_THREADS
  try {
    ids = obj->impl->encode(std::string(utf8, static_cast<size_t>(len)),
                            static_cast<size_t>(chunk_size), num_threads);
  } catch (const std::exception& e) {
    error = e.what();
    failed = true;
  }
  Py_END_ALLOW_THREADS
  if (failed) {
    PyErr_SetString(PyExc_RuntimeError, error.c_str());
    return nullptr;
  }
  PyObject* list = PyList_New(static_cast<Py_ssize_t>(ids.size()));
  if (list == nullptr) return nullptr;
  for (size_t i = 0; i < ids.size(); ++i) {
    PyObject* item = PyLong_FromLong(static_cast<long>(ids[i]));
    if (item == nullptr) {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SetItem(list, static_cast<Py_ssize_t>(i), item);
  }
  return list;
}

PyMethodDef tokenizer_methods[] = {
    {"encode", reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(Tokenizer_encode)),
     METH_VARARGS | METH_KEYWORDS, "encode(text, chunk_size=16384, num_threads=0) -> list[int]"},
    {nullptr, nullptr, 0, nullptr},
};

PyType_Slot tokenizer_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyType_GenericNew)},
    {Py_tp_init, reinterpret_cast<void*>(Tokenizer_init)},
    {Py_tp_dealloc, reinterpret_cast<void*>(Tokenizer_dealloc)},
    {Py_tp_methods, reinterpret_cast<void*>(tokenizer_methods)},
    {0, nullptr},
};

PyType_Spec tokenizer_spec = {
    "sentencepiece_lite.FastSBPTokenizer",
    sizeof(TokenizerObject),
    0,
    Py_TPFLAGS_DEFAULT,
    tokenizer_slots,
};

PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "sentencepiece_lite", nullptr, -1,
                          nullptr, nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit_sentencepiece_lite(void) {
  PyObject* type = PyType_FromSpec(&tokenizer_spec);
  if (type == nullptr) return nullptr;
  PyObject* module = PyModule_Create(&module_def);
  if (module == nullptr) {
    Py_DECREF(type);
    return nullptr;
  }
  if (PyModule_AddObjectRef(module, "FastSBPTokenizer", type) < 0) {
    Py_DECREF(type);
    Py_DECREF(module);
    return nullptr;
  }
  Py_DECREF(type);
  return module;
}
