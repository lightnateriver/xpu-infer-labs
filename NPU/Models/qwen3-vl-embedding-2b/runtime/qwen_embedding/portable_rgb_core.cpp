#include <cstddef>
#include <cstdint>

// Integer color arithmetic matching the documented FFmpeg SSSE3 operation order.
// No CPU-specific intrinsics and no floating-point intermediates.
static_assert((-1 >> 1) == -1, "Arithmetic signed shift required");
static inline uint8_t clip_byte(int32_t value) {
    return static_cast<uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

extern "C" int rgb420_compat(const uint8_t* input, uint8_t* output,
                              size_t frames, int height, int width,
                              int cy, int offset, int cr, int cb, int cgu, int cgv) {
    if (!input || !output || !frames || height <= 0 || width <= 0 || height % 2 || width % 2) return 1;
    const size_t plane = static_cast<size_t>(height) * width;
    for (size_t frame = 0; frame < frames; ++frame) {
        const uint8_t* y = input + frame * (plane + plane / 2);
        const uint8_t* u = y + plane;
        const uint8_t* v = u + plane / 4;
        uint8_t* dst = output + frame * plane * 3;
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                const size_t i = static_cast<size_t>(row) * width + column;
                const size_t chroma = static_cast<size_t>(row / 2) * (width / 2) + column / 2;
                const int32_t yy = ((8 * static_cast<int32_t>(y[i]) - offset) * cy) >> 16;
                const int32_t uu = (static_cast<int32_t>(u[chroma]) - 128) * 8;
                const int32_t vv = (static_cast<int32_t>(v[chroma]) - 128) * 8;
                dst[i * 3] = clip_byte(yy + ((vv * cr) >> 16));
                dst[i * 3 + 1] = clip_byte(yy + ((uu * cgu) >> 16) + ((vv * cgv) >> 16));
                dst[i * 3 + 2] = clip_byte(yy + ((uu * cb) >> 16));
            }
        }
    }
    return 0;
}
