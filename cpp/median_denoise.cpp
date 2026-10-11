// 3x3 median filter for 8-bit binary PGM images.
// Python owns JPEG/PNG decoding and writes canonical P5 files. This program
// is the denoising step invoked from the Python pipeline.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

std::string next_token(std::istream& in) {
    std::string token;
    char ch = 0;
    while (in.get(ch)) {
        if (ch == '#') {
            while (in.get(ch) && ch != '\n') {
            }
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
            break;
        }
    }
    while (in.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            break;
        }
        token.push_back(ch);
    }
    return token;
}

bool read_pgm(const fs::path& path, Image& image) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[2] = {};
    in.read(magic, 2);
    if (magic[0] != 'P' || magic[1] != '5') {
        return false;
    }
    image.width = std::stoi(next_token(in));
    image.height = std::stoi(next_token(in));
    const int max_value = std::stoi(next_token(in));
    if (image.width <= 0 || image.height <= 0 || max_value != 255) {
        return false;
    }
    image.pixels.resize(static_cast<size_t>(image.width) * static_cast<size_t>(image.height));
    in.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    return in.gcount() == static_cast<std::streamsize>(image.pixels.size());
}

bool write_pgm(const fs::path& path, const Image& image) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P5\n" << image.width << " " << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    return static_cast<bool>(out);
}

uint8_t median_at(const Image& source, int x, int y) {
    uint8_t window[9];
    int cursor = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = std::clamp(y + dy, 0, source.height - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = std::clamp(x + dx, 0, source.width - 1);
            window[cursor++] = source.pixels[static_cast<size_t>(yy) * source.width + xx];
        }
    }
    std::nth_element(window, window + 4, window + 9);
    return window[4];
}

Image median3(const Image& source) {
    Image output;
    output.width = source.width;
    output.height = source.height;
    output.pixels.resize(source.pixels.size());
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            output.pixels[static_cast<size_t>(y) * source.width + x] = median_at(source, x, y);
        }
    }
    return output;
}

std::vector<fs::path> arguments() {
    std::vector<fs::path> values;
#ifdef _WIN32
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argv == nullptr) {
        return values;
    }
    for (int i = 0; i < argc; ++i) {
        values.emplace_back(argv[i]);
    }
    LocalFree(argv);
#else
    // Non-Windows builds receive UTF-8 argv from main().
#endif
    return values;
}

int run(const fs::path& input_dir, const fs::path& output_dir) {
    if (!fs::exists(input_dir)) {
        std::cerr << "input directory not found\n";
        return 1;
    }
    fs::create_directories(output_dir);
    int count = 0;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != fs::path(".pgm")) {
            continue;
        }
        Image image;
        if (!read_pgm(entry.path(), image)) {
            std::cerr << "failed to read " << entry.path().filename().string() << "\n";
            return 1;
        }
        const Image denoised = median3(image);
        if (!write_pgm(output_dir / entry.path().filename(), denoised)) {
            std::cerr << "failed to write " << entry.path().filename().string() << "\n";
            return 1;
        }
        ++count;
    }
    std::cout << "denoised " << count << " images\n";
    return 0;
}

#ifdef _WIN32
int main() {
    const std::vector<fs::path> args = arguments();
    if (args.size() != 3) {
        std::cerr << "usage: median_denoise <input_pgm_dir> <output_pgm_dir>\n";
        return 2;
    }
    return run(args[1], args[2]);
}
#else
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: median_denoise <input_pgm_dir> <output_pgm_dir>\n";
        return 2;
    }
    return run(fs::path(argv[1]), fs::path(argv[2]));
}
#endif
