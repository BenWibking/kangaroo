#include "flux_geometry_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace kangaroo {

double min_dist_sq_to_interval(double a0, double a1) {
  if (a1 < 0.0) {
    return a1 * a1;
  }
  if (a0 > 0.0) {
    return a0 * a0;
  }
  return 0.0;
}

double max_dist_sq_to_interval(double a0, double a1) {
  const double aa0 = std::abs(a0);
  const double aa1 = std::abs(a1);
  const double amax = (aa0 > aa1) ? aa0 : aa1;
  return amax * amax;
}

bool sphere_may_intersect_cell(double radius2, double x0, double x1, double y0,
                               double y1, double z0, double z1) {
  const double r2_min = min_dist_sq_to_interval(x0, x1) +
                        min_dist_sq_to_interval(y0, y1) +
                        min_dist_sq_to_interval(z0, z1);
  const double r2_max = max_dist_sq_to_interval(x0, x1) +
                        max_dist_sq_to_interval(y0, y1) +
                        max_dist_sq_to_interval(z0, z1);
  return radius2 >= r2_min && radius2 <= r2_max;
}

bool cylinder_may_intersect_cell(double radius2, double height, double x0,
                                 double x1, double y0, double y1, double z0,
                                 double z1) {
  const double r2_min =
      min_dist_sq_to_interval(x0, x1) + min_dist_sq_to_interval(y0, y1);
  const double r2_max =
      max_dist_sq_to_interval(x0, x1) + max_dist_sq_to_interval(y0, y1);
  return radius2 >= r2_min && radius2 <= r2_max && z1 >= -height &&
         z0 <= height;
}

CellIndexRange cells_intersecting_axis_band(double band_lo, double band_hi,
                                            double x0, double dx,
                                            int32_t index_origin,
                                            int32_t block_lo,
                                            int32_t block_hi) {
  if (!std::isfinite(band_lo) || !std::isfinite(band_hi) ||
      !std::isfinite(x0) || !std::isfinite(dx) || dx <= 0.0 ||
      band_hi < band_lo || block_hi < block_lo) {
    return {};
  }

  const double scaled_first =
      (band_lo - x0) / dx + static_cast<double>(index_origin) - 1.0;
  const double scaled_last =
      (band_hi - x0) / dx + static_cast<double>(index_origin);
  constexpr double pad = 1.0e-12;
  const double raw_first = std::ceil(scaled_first - pad) - 1.0;
  const double raw_last = std::floor(scaled_last + pad) + 1.0;
  const double block_lo_value = static_cast<double>(block_lo);
  const double block_hi_value = static_cast<double>(block_hi);
  if (raw_first > block_hi_value || raw_last < block_lo_value) {
    return {};
  }
  const int64_t clamped_first = static_cast<int64_t>(
      std::max(block_lo_value, raw_first));
  const int64_t clamped_last = static_cast<int64_t>(
      std::min(block_hi_value, raw_last));
  if (clamped_first > clamped_last) {
    return {};
  }
  return CellIndexRange{static_cast<int>(clamped_first - block_lo),
                        static_cast<int>(clamped_last - block_lo)};
}

namespace {

struct FluxPoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

void add_point_unique(std::array<FluxPoint, 16> &pts, int &npts, double x,
                      double y, double z, double tol) {
  const double tol2 = tol * tol;
  for (int i = 0; i < npts; ++i) {
    const double dx = pts[static_cast<std::size_t>(i)].x - x;
    const double dy = pts[static_cast<std::size_t>(i)].y - y;
    const double dz = pts[static_cast<std::size_t>(i)].z - z;
    if ((dx * dx + dy * dy + dz * dz) <= tol2) {
      return;
    }
  }
  if (npts < static_cast<int>(pts.size())) {
    pts[static_cast<std::size_t>(npts)] = FluxPoint{x, y, z};
    ++npts;
  }
}

} // namespace

double plane_box_section_area(double x0, double x1, double y0, double y1,
                              double z0, double z1, double nx, double ny,
                              double nz, double d) {
  const double scale = std::abs(x0) + std::abs(x1) + std::abs(y0) +
                       std::abs(y1) + std::abs(z0) + std::abs(z1) +
                       std::abs(d) + 1.0;
  const double tol = 1.0e-12 * scale;

  const std::array<FluxPoint, 8> verts{
      FluxPoint{x0, y0, z0}, FluxPoint{x1, y0, z0}, FluxPoint{x0, y1, z0},
      FluxPoint{x1, y1, z0}, FluxPoint{x0, y0, z1}, FluxPoint{x1, y0, z1},
      FluxPoint{x0, y1, z1}, FluxPoint{x1, y1, z1}};
  const std::array<std::array<int, 2>, 12> edges{
      std::array<int, 2>{0, 1}, std::array<int, 2>{2, 3},
      std::array<int, 2>{4, 5}, std::array<int, 2>{6, 7},
      std::array<int, 2>{0, 2}, std::array<int, 2>{1, 3},
      std::array<int, 2>{4, 6}, std::array<int, 2>{5, 7},
      std::array<int, 2>{0, 4}, std::array<int, 2>{1, 5},
      std::array<int, 2>{2, 6}, std::array<int, 2>{3, 7}};

  std::array<FluxPoint, 16> pts{};
  int npts = 0;

  for (const auto &edge : edges) {
    const int i0 = edge[0];
    const int i1 = edge[1];
    const auto &p0 = verts[static_cast<std::size_t>(i0)];
    const auto &p1 = verts[static_cast<std::size_t>(i1)];

    double f0 = nx * p0.x + ny * p0.y + nz * p0.z - d;
    double f1 = nx * p1.x + ny * p1.y + nz * p1.z - d;
    if (std::abs(f0) <= tol) {
      f0 = 0.0;
    }
    if (std::abs(f1) <= tol) {
      f1 = 0.0;
    }

    if (f0 == 0.0 && f1 == 0.0) {
      add_point_unique(pts, npts, p0.x, p0.y, p0.z, tol);
      add_point_unique(pts, npts, p1.x, p1.y, p1.z, tol);
      continue;
    }
    if (f0 == 0.0) {
      add_point_unique(pts, npts, p0.x, p0.y, p0.z, tol);
      continue;
    }
    if (f1 == 0.0) {
      add_point_unique(pts, npts, p1.x, p1.y, p1.z, tol);
      continue;
    }
    if ((f0 < 0.0 && f1 > 0.0) || (f0 > 0.0 && f1 < 0.0)) {
      const double t = f0 / (f0 - f1);
      const double x = p0.x + t * (p1.x - p0.x);
      const double y = p0.y + t * (p1.y - p0.y);
      const double z = p0.z + t * (p1.z - p0.z);
      add_point_unique(pts, npts, x, y, z, tol);
    }
  }

  if (npts < 3) {
    return 0.0;
  }

  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  for (int i = 0; i < npts; ++i) {
    cx += pts[static_cast<std::size_t>(i)].x;
    cy += pts[static_cast<std::size_t>(i)].y;
    cz += pts[static_cast<std::size_t>(i)].z;
  }
  cx /= static_cast<double>(npts);
  cy /= static_cast<double>(npts);
  cz /= static_cast<double>(npts);

  double ax = 1.0;
  double ay = 0.0;
  double az = 0.0;
  if (std::abs(nx) > 0.9) {
    ax = 0.0;
    ay = 1.0;
    az = 0.0;
  }
  double e1x = ny * az - nz * ay;
  double e1y = nz * ax - nx * az;
  double e1z = nx * ay - ny * ax;
  const double e1norm = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
  if (e1norm <= 0.0) {
    return 0.0;
  }
  e1x /= e1norm;
  e1y /= e1norm;
  e1z /= e1norm;

  const double e2x = ny * e1z - nz * e1y;
  const double e2y = nz * e1x - nx * e1z;
  const double e2z = nx * e1y - ny * e1x;

  std::array<double, 16> u{};
  std::array<double, 16> v{};
  std::array<double, 16> ang{};
  for (int i = 0; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double rx = pts[idx].x - cx;
    const double ry = pts[idx].y - cy;
    const double rz = pts[idx].z - cz;
    u[idx] = rx * e1x + ry * e1y + rz * e1z;
    v[idx] = rx * e2x + ry * e2y + rz * e2z;
    ang[idx] = std::atan2(v[idx], u[idx]);
  }

  for (int i = 1; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double key_ang = ang[idx];
    const double key_u = u[idx];
    const double key_v = v[idx];
    int j = i - 1;
    while (j >= 0 && ang[static_cast<std::size_t>(j)] > key_ang) {
      const std::size_t dst = static_cast<std::size_t>(j + 1);
      const std::size_t src = static_cast<std::size_t>(j);
      ang[dst] = ang[src];
      u[dst] = u[src];
      v[dst] = v[src];
      --j;
    }
    const std::size_t dst = static_cast<std::size_t>(j + 1);
    ang[dst] = key_ang;
    u[dst] = key_u;
    v[dst] = key_v;
  }

  double area2 = 0.0;
  for (int i = 0; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const std::size_t next =
        static_cast<std::size_t>((i + 1 < npts) ? (i + 1) : 0);
    area2 += u[idx] * v[next] - v[idx] * u[next];
  }
  return 0.5 * std::abs(area2);
}

double spherical_section_area_in_intersecting_cell(double radius, double x0,
                                                   double x1, double y0,
                                                   double y1, double z0,
                                                   double z1) {
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const double dz = z1 - z0;
  const double vol = dx * dy * dz;
  if (vol <= 0.0) {
    return 0.0;
  }

  const double xc = 0.5 * (x0 + x1);
  const double yc = 0.5 * (y0 + y1);
  const double zc = 0.5 * (z0 + z1);
  const double rc = std::sqrt(xc * xc + yc * yc + zc * zc);
  if (rc <= 0.0) {
    return 0.0;
  }

  return plane_box_section_area(x0, x1, y0, y1, z0, z1, xc / rc, yc / rc,
                                zc / rc, radius);
}

namespace {

double spherical_section_area_in_cell(double radius, double x0, double x1,
                                      double y0, double y1, double z0,
                                      double z1) {
  const double radius2 = radius * radius;
  if (!sphere_may_intersect_cell(radius2, x0, x1, y0, y1, z0, z1)) {
    return 0.0;
  }
  return spherical_section_area_in_intersecting_cell(radius, x0, x1, y0, y1, z0,
                                                     z1);
}

} // namespace

double cylindrical_section_area_in_intersecting_cell(double radius,
                                                     double height, double x0,
                                                     double x1, double y0,
                                                     double y1, double z0,
                                                     double z1) {
  const double zlo = std::max(z0, -height);
  const double zhi = std::min(z1, height);
  if (zhi <= zlo) {
    return 0.0;
  }

  const double xc = 0.5 * (x0 + x1);
  const double yc = 0.5 * (y0 + y1);
  const double rc = std::sqrt(xc * xc + yc * yc);
  if (rc <= 0.0) {
    return 0.0;
  }

  return plane_box_section_area(x0, x1, y0, y1, zlo, zhi, xc / rc, yc / rc, 0.0,
                                radius);
}

} // namespace kangaroo
