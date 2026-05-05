#include "../../../../reference/pocketfft-upstream/pocketfft_hdronly.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
using pocketfft::shape_t;
using pocketfft::stride_t;

std::ptrdiff_t real_stride(long stride) {
  return static_cast<std::ptrdiff_t>(stride * static_cast<long>(sizeof(double)));
}

std::vector<double> filled(std::size_t size, double value) {
  return std::vector<double>(size, value);
}

std::vector<std::complex<double>> complex_filled(std::size_t size, double real, double imaginary) {
  return std::vector<std::complex<double>>(size, {real, imaginary});
}

std::vector<double> real_sample(int size) {
  std::vector<double> data(static_cast<std::size_t>(size));
  for (int index = 0; index < size; ++index) {
    data[static_cast<std::size_t>(index)] = 0.25 + 0.125 * ((13 * index + 7) % 17)
        - 0.03125 * (index % 5);
  }
  return data;
}

std::vector<std::complex<double>> complex_sample(int size) {
  std::vector<std::complex<double>> data(static_cast<std::size_t>(size));
  for (int index = 0; index < size; ++index) {
    double real = 0.25 + 0.125 * ((11 * index + 3) % 19);
    double imaginary = -0.5 + 0.0625 * ((7 * index + 5) % 23);
    data[static_cast<std::size_t>(index)] = {real, imaginary};
  }
  return data;
}

std::vector<double> real_grid(int rows, int columns) {
  std::vector<double> data(static_cast<std::size_t>(rows * columns));
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      data[static_cast<std::size_t>(row * columns + column)] = 0.25 + 0.5 * row - 0.125 * column
          + 0.03125 * ((row + 3 * column) % 5);
    }
  }
  return data;
}

std::vector<double> real_cube(int planes, int rows, int columns) {
  std::vector<double> data(static_cast<std::size_t>(planes * rows * columns));
  for (int plane = 0; plane < planes; ++plane) {
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        int offset = (plane * rows + row) * columns + column;
        data[static_cast<std::size_t>(offset)] = 0.1875 + 0.25 * plane - 0.15625 * row
            + 0.09375 * column + 0.015625 * ((5 * plane + 7 * row + 11 * column) % 13);
      }
    }
  }
  return data;
}

std::vector<std::complex<double>> complex_grid(int rows, int columns) {
  std::vector<std::complex<double>> data(static_cast<std::size_t>(rows * columns));
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      double real = 0.25 + 0.5 * row + 0.125 * ((7 * column + row) % 11);
      double imaginary = -0.375 + 0.0625 * ((5 * row + 3 * column) % 13);
      data[static_cast<std::size_t>(row * columns + column)] = {real, imaginary};
    }
  }
  return data;
}

std::vector<std::complex<double>> complex_cube(int planes, int rows, int columns) {
  std::vector<std::complex<double>> data(static_cast<std::size_t>(planes * rows * columns));
  for (int plane = 0; plane < planes; ++plane) {
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        double real = 0.125 + 0.375 * plane + 0.25 * row - 0.0625 * column
            + 0.03125 * ((3 * plane + 5 * row + 7 * column) % 11);
        double imaginary = -0.25 + 0.125 * plane - 0.03125 * row
            + 0.0625 * ((5 * plane + row + 3 * column) % 13);
        int offset = (plane * rows + row) * columns + column;
        data[static_cast<std::size_t>(offset)] = {real, imaginary};
      }
    }
  }
  return data;
}

std::vector<double> strided_real_sample(int size) {
  auto compact = real_sample(size);
  auto data = filled(static_cast<std::size_t>(1 + (size - 1) * 2 + 1), -1111.0);
  for (int index = 0; index < size; ++index) {
    data[static_cast<std::size_t>(1 + 2 * index)] = compact[static_cast<std::size_t>(index)];
  }
  return data;
}

std::vector<std::complex<double>> strided_complex_sample(int complex_length) {
  auto compact = complex_sample(complex_length);
  auto data = complex_filled(static_cast<std::size_t>(1 + (complex_length - 1) * 2 + 1), -555.0, 555.0);
  for (int index = 0; index < complex_length; ++index) {
    data[static_cast<std::size_t>(1 + 2 * index)] = compact[static_cast<std::size_t>(index)];
  }
  return data;
}

std::vector<std::complex<double>> strided_complex_signal(int size, int offset, int stride, double real,
                                                         double imaginary) {
  auto compact = complex_sample(size);
  auto data = complex_filled(static_cast<std::size_t>(offset + (size - 1) * stride + 1), real, imaginary);
  for (int index = 0; index < size; ++index) {
    data[static_cast<std::size_t>(offset + stride * index)] = compact[static_cast<std::size_t>(index)];
  }
  return data;
}

void print_real_array(const std::string &name, const std::vector<double> &data) {
  std::cout << "real\t" << name;
  for (double value : data) {
    std::cout << '\t' << std::setprecision(17) << value;
  }
  std::cout << '\n';
}

void print_complex_array(const std::string &name, const std::vector<std::complex<double>> &data) {
  std::cout << "complex\t" << name;
  for (const auto &value : data) {
    std::cout << '\t' << std::setprecision(17) << value.real()
              << '\t' << std::setprecision(17) << value.imag();
  }
  std::cout << '\n';
}

void emit_1d_complex() {
  for (int n : {1, 2, 3, 4, 5, 8, 13, 77, 257}) {
    shape_t shape{static_cast<std::size_t>(n)};
    stride_t stride{real_stride(2)};
    auto input = complex_sample(n);
    auto output = input;
    pocketfft::c2c(shape, stride, stride, shape_t{0}, pocketfft::FORWARD, input.data(), output.data(), 1.0, 1);
    print_complex_array("c2c_forward_" + std::to_string(n), output);

    auto backward = input;
    pocketfft::c2c(shape, stride, stride, shape_t{0}, pocketfft::BACKWARD, backward.data(), backward.data(), 1.0,
                   1);
    print_complex_array("c2c_backward_" + std::to_string(n), backward);

    pocketfft::c2c(shape, stride, stride, shape_t{0}, pocketfft::BACKWARD, output.data(), output.data(),
                   1.0 / static_cast<double>(n), 1);
    print_complex_array("c2c_roundtrip_" + std::to_string(n), output);
  }
}

void emit_1d_real_complex() {
  for (int n : {1, 2, 3, 4, 5, 8, 17, 77, 257}) {
    shape_t shape{static_cast<std::size_t>(n)};
    stride_t real_stride_unit{real_stride(1)};
    stride_t complex_stride_unit{real_stride(2)};
    auto input = real_sample(n);
    std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(n / 2 + 1));
    pocketfft::r2c(shape, real_stride_unit, complex_stride_unit, 0, pocketfft::FORWARD, input.data(), spectrum.data(),
                   1.0, 1);
    print_complex_array("r2c_forward_" + std::to_string(n), spectrum);

    std::vector<std::complex<double>> backward_spectrum(static_cast<std::size_t>(n / 2 + 1));
    pocketfft::r2c(shape, real_stride_unit, complex_stride_unit, 0, pocketfft::BACKWARD, input.data(),
                   backward_spectrum.data(), 1.0, 1);
    print_complex_array("r2c_backward_" + std::to_string(n), backward_spectrum);

    std::vector<double> roundtrip(static_cast<std::size_t>(n));
    pocketfft::c2r(shape, complex_stride_unit, real_stride_unit, 0, pocketfft::BACKWARD, spectrum.data(),
                   roundtrip.data(), 1.0 / static_cast<double>(n), 1);
    print_real_array("c2r_roundtrip_" + std::to_string(n), roundtrip);
  }
}

void emit_1d_real_families() {
  for (int n : {1, 2, 3, 4, 5, 8, 17}) {
    shape_t shape{static_cast<std::size_t>(n)};
    stride_t stride{real_stride(1)};
    auto input = real_sample(n);

    auto output = input;
    pocketfft::r2r_fftpack(shape, stride, stride, shape_t{0}, true, true, input.data(), output.data(), 1.0, 1);
    print_real_array("r2r_fftpack_forward_" + std::to_string(n), output);

    output = input;
    pocketfft::r2r_separable_hartley(shape, stride, stride, shape_t{0}, input.data(), output.data(), 1.0, 1);
    print_real_array("hartley_" + std::to_string(n), output);

    output = input;
    pocketfft::r2r_separable_fht(shape, stride, stride, shape_t{0}, input.data(), output.data(), 1.0, 1);
    print_real_array("fht_" + std::to_string(n), output);

    for (int type = 1; type <= 4; ++type) {
      if (type == 1 && n < 2) {
        continue;
      }
      output = input;
      pocketfft::dct(shape, stride, stride, shape_t{0}, type, input.data(), output.data(), 1.0, false, 1);
      print_real_array("dct" + std::to_string(type) + "_" + std::to_string(n), output);

      output = input;
      pocketfft::dst(shape, stride, stride, shape_t{0}, type, input.data(), output.data(), 1.0, false, 1);
      print_real_array("dst" + std::to_string(type) + "_" + std::to_string(n), output);
    }
  }
}

void emit_nd() {
  shape_t shape{3, 4};
  stride_t real_stride_2d{real_stride(4), real_stride(1)};
  stride_t complex_stride_2d{real_stride(8), real_stride(2)};
  stride_t spectrum_stride_2d{real_stride(6), real_stride(2)};
  shape_t axes{0, 1};

  auto complex_input = complex_grid(3, 4);
  std::vector<std::complex<double>> complex_output(complex_input.size());
  pocketfft::c2c(shape, complex_stride_2d, complex_stride_2d, axes, pocketfft::FORWARD, complex_input.data(),
                 complex_output.data(), 0.5, 1);
  print_complex_array("c2c_2d_forward_3x4", complex_output);

  auto real_input = real_grid(3, 4);
  std::vector<std::complex<double>> spectrum(3 * (4 / 2 + 1));
  pocketfft::r2c(shape, real_stride_2d, spectrum_stride_2d, axes, pocketfft::FORWARD, real_input.data(),
                 spectrum.data(), 1.0, 1);
  print_complex_array("r2c_2d_forward_3x4", spectrum);

  std::vector<double> restored(real_input.size());
  pocketfft::c2r(shape, spectrum_stride_2d, real_stride_2d, axes, pocketfft::BACKWARD, spectrum.data(),
                 restored.data(), 1.0 / 12.0, 1);
  print_real_array("c2r_2d_roundtrip_3x4", restored);

  std::vector<double> real_output(real_input.size());
  pocketfft::r2r_genuine_hartley(shape, real_stride_2d, real_stride_2d, axes, real_input.data(), real_output.data(),
                                 1.0, 1);
  print_real_array("genuine_hartley_2d_3x4", real_output);

  pocketfft::r2r_genuine_fht(shape, real_stride_2d, real_stride_2d, axes, real_input.data(), real_output.data(), 1.0,
                             1);
  print_real_array("genuine_fht_2d_3x4", real_output);

  pocketfft::dct(shape, real_stride_2d, real_stride_2d, shape_t{1}, 2, real_input.data(), real_output.data(), 0.75,
                 false, 1);
  print_real_array("dct2_axis1_3x4", real_output);
}

void emit_negative_stride() {
  shape_t shape{5};
  shape_t axes{0};
  auto real_input = real_sample(5);
  std::vector<double> real_output(5);
  pocketfft::r2r_fftpack(shape, stride_t{real_stride(-1)}, stride_t{real_stride(1)}, axes, true, true,
                         real_input.data() + 4, real_output.data(), 1.0, 1);
  print_real_array("r2r_fftpack_negative_stride_5", real_output);

  auto complex_input = complex_sample(5);
  std::vector<std::complex<double>> complex_output(complex_input.size());
  pocketfft::c2c(shape, stride_t{real_stride(-2)}, stride_t{real_stride(2)}, axes, pocketfft::FORWARD,
                 complex_input.data() + 4, complex_output.data(), 1.0, 1);
  print_complex_array("c2c_negative_stride_5", complex_output);
}

void emit_real_complex_strided() {
  shape_t axes{0};
  auto r2c_input6 = strided_real_sample(6);
  auto r2c_forward6 = complex_filled(1 + (6 / 2) * 2 + 1, -777.0, 777.0);
  pocketfft::r2c(shape_t{6}, stride_t{real_stride(2)}, stride_t{real_stride(4)}, 0, pocketfft::FORWARD,
                 r2c_input6.data() + 1, r2c_forward6.data() + 1, 0.75, 1);
  print_complex_array("r2c_strided_forward_6", r2c_forward6);

  auto r2c_input7 = strided_real_sample(7);
  auto r2c_backward7 = complex_filled(1 + (7 / 2) * 2 + 1, -777.0, 777.0);
  pocketfft::r2c(shape_t{7}, stride_t{real_stride(2)}, stride_t{real_stride(4)}, 0, pocketfft::BACKWARD,
                 r2c_input7.data() + 1, r2c_backward7.data() + 1, 1.25, 1);
  print_complex_array("r2c_strided_backward_7", r2c_backward7);

  auto c2r_input6 = strided_complex_sample(6 / 2 + 1);
  auto c2r_forward6 = filled(3 + (6 - 1) * 2 + 1, -333.0);
  pocketfft::c2r(shape_t{6}, stride_t{real_stride(4)}, stride_t{real_stride(2)}, 0, pocketfft::FORWARD,
                 c2r_input6.data() + 1, c2r_forward6.data() + 3, 0.5, 1);
  print_real_array("c2r_strided_forward_6", c2r_forward6);

  auto c2r_input7 = strided_complex_sample(7 / 2 + 1);
  auto c2r_backward7 = filled(3 + (7 - 1) * 2 + 1, -333.0);
  pocketfft::c2r(shape_t{7}, stride_t{real_stride(4)}, stride_t{real_stride(2)}, 0, pocketfft::BACKWARD,
                 c2r_input7.data() + 1, c2r_backward7.data() + 3, 0.875, 1);
  print_real_array("c2r_strided_backward_7", c2r_backward7);
}

void emit_layout_sensitive_real_families() {
  shape_t shape{3, 4};
  stride_t real_stride_2d{real_stride(4), real_stride(1)};
  stride_t column_major_stride{real_stride(1), real_stride(3)};
  auto real_input = real_grid(3, 4);

  std::vector<double> values(12, -222.0);
  pocketfft::r2r_separable_hartley(shape, real_stride_2d, column_major_stride, shape_t{0}, real_input.data(),
                                   values.data(), 1.25, 1);
  print_real_array("hartley_axis0_colmajor_3x4", values);

  std::fill(values.begin(), values.end(), -222.0);
  pocketfft::r2r_separable_fht(shape, real_stride_2d, column_major_stride, shape_t{0}, real_input.data(),
                               values.data(), 1.25, 1);
  print_real_array("fht_axis0_colmajor_3x4", values);

  std::vector<double> genuine_hartley(20, -9999.0);
  stride_t offset_stride{real_stride(5), real_stride(1)};
  pocketfft::r2r_genuine_hartley(shape, real_stride_2d, offset_stride, shape_t{0, 1}, real_input.data(),
                                 genuine_hartley.data() + 2, 0.75, 1);
  print_real_array("genuine_hartley_strided_offset_3x4", genuine_hartley);

  std::vector<double> genuine_fht(20, -9999.0);
  pocketfft::r2r_genuine_fht(shape, real_stride_2d, offset_stride, shape_t{0, 1}, real_input.data(),
                             genuine_fht.data() + 2, 0.75, 1);
  print_real_array("genuine_fht_strided_offset_3x4", genuine_fht);

  shape_t dcst_shape{4, 3};
  stride_t dcst_in_stride{real_stride(5), real_stride(1)};
  stride_t dcst_out_stride{real_stride(7), real_stride(2)};
  auto dcst_input = filled(32, -4444.0);
  auto dcst_output = filled(40, -9999.0);
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 3; ++column) {
      dcst_input[static_cast<std::size_t>(2 + row * 5 + column)] = 0.25 + 0.5 * row - 0.125 * column;
    }
  }
  pocketfft::dst(dcst_shape, dcst_in_stride, dcst_out_stride, shape_t{0}, 3, dcst_input.data() + 2,
                 dcst_output.data() + 4, 0.5, false, 1);
  print_real_array("dst3_axis0_strided_offset_4x3", dcst_output);
}

void emit_strided_complex() {
  shape_t axes{0};
  auto input6 = strided_complex_signal(6, 2, 2, -901.0, 901.0);
  auto output6 = complex_filled(1 + (6 - 1) * 3 + 1, -902.0, 902.0);
  pocketfft::c2c(shape_t{6}, stride_t{real_stride(4)}, stride_t{real_stride(6)}, axes, pocketfft::FORWARD,
                 input6.data() + 2, output6.data() + 1, 0.625, 1);
  print_complex_array("c2c_strided_forward_6", output6);

  auto input7 = strided_complex_signal(7, 1, 3, -903.0, 903.0);
  auto output7 = complex_filled(2 + (7 - 1) * 2 + 1, -904.0, 904.0);
  pocketfft::c2c(shape_t{7}, stride_t{real_stride(6)}, stride_t{real_stride(4)}, axes, pocketfft::BACKWARD,
                 input7.data() + 1, output7.data() + 2, 0.375, 1);
  print_complex_array("c2c_strided_backward_7", output7);
}

void emit_nd_real_complex_axes() {
  shape_t shape{2, 3, 4};
  shape_t axes{0, 2};
  stride_t real_stride_3d{real_stride(12), real_stride(4), real_stride(1)};
  stride_t spectrum_stride_3d{real_stride(18), real_stride(6), real_stride(2)};
  auto input = real_cube(2, 3, 4);
  std::vector<std::complex<double>> spectrum(2 * 3 * (4 / 2 + 1));
  pocketfft::r2c(shape, real_stride_3d, spectrum_stride_3d, axes, pocketfft::FORWARD, input.data(), spectrum.data(),
                 0.5, 1);
  print_complex_array("r2c_3d_axes_0_2_2x3x4", spectrum);

  std::vector<double> restored(input.size());
  pocketfft::c2r(shape, spectrum_stride_3d, real_stride_3d, axes, pocketfft::BACKWARD, spectrum.data(),
                 restored.data(), 0.25, 1);
  print_real_array("c2r_3d_axes_0_2_roundtrip_2x3x4", restored);
}

void emit_dcst_ortho_and_fftpack_h2r() {
  auto dct_input = real_sample(8);
  auto dct_output = dct_input;
  pocketfft::dct(shape_t{8}, stride_t{real_stride(1)}, stride_t{real_stride(1)}, shape_t{0}, 2, dct_input.data(),
                 dct_output.data(), 0.5, true, 1);
  print_real_array("dct2_ortho_factor_8", dct_output);

  auto dst_input = real_sample(7);
  auto dst_output = dst_input;
  pocketfft::dst(shape_t{7}, stride_t{real_stride(1)}, stride_t{real_stride(1)}, shape_t{0}, 4, dst_input.data(),
                 dst_output.data(), 1.25, true, 1);
  print_real_array("dst4_ortho_factor_7", dst_output);

  auto h2r_input = strided_real_sample(8);
  auto h2r_output = filled(2 + (8 - 1) * 3 + 1, -444.0);
  pocketfft::r2r_fftpack(shape_t{8}, stride_t{real_stride(2)}, stride_t{real_stride(3)}, shape_t{0}, false, false,
                         h2r_input.data() + 1, h2r_output.data() + 2, 0.625, 1);
  print_real_array("r2r_fftpack_h2r_strided_backward_8", h2r_output);
}

void emit_nd_complex_axes() {
  shape_t shape{2, 3, 5};
  stride_t stride{real_stride(30), real_stride(10), real_stride(2)};
  auto input = complex_cube(2, 3, 5);
  std::vector<std::complex<double>> output(input.size());
  pocketfft::c2c(shape, stride, stride, shape_t{2, 0}, pocketfft::FORWARD, input.data(), output.data(), 0.25, 1);
  print_complex_array("c2c_nd_axes_2_0_factor_2x3x5", output);
}

}  // namespace

int main() {
  emit_1d_complex();
  emit_1d_real_complex();
  emit_1d_real_families();
  emit_nd();
  emit_negative_stride();
  emit_real_complex_strided();
  emit_layout_sensitive_real_families();
  emit_strided_complex();
  emit_nd_real_complex_axes();
  emit_dcst_ortho_and_fftpack_h2r();
  emit_nd_complex_axes();
  return 0;
}
