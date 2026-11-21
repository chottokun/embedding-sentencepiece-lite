#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

namespace py = pybind11;

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

PYBIND11_MODULE(sentencepiece_lite, m) {
  m.doc() = "SentencePiece Lite Python Wrapper with Fast SBP Tokenization";
  py::class_<FastSBPTokenizer>(m, "FastSBPTokenizer")
      .def(py::init<const std::string&>(), py::arg("model_path"))
      .def("encode", &FastSBPTokenizer::encode,
           py::arg("text"), py::arg("chunk_size") = 16384, py::arg("num_threads") = 0);
}
