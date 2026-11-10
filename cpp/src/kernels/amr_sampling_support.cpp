#include "amr_sampling_support.hpp"

#include <algorithm>
#include <cmath>

namespace kangaroo {

double cell_edge(const LevelGeom &geom, int axis, int idx) {
  return geom.x0[axis] + (idx - geom.index_origin[axis]) * geom.dx[axis];
}

double cell_center(const LevelGeom &geom, int axis, int idx) {
  return cell_edge(geom, axis, idx) + 0.5 * geom.dx[axis];
}

int32_t coord_to_index(const LevelGeom &geom, int axis, double x) {
  return static_cast<int32_t>(std::floor((x - geom.x0[axis]) / geom.dx[axis])) +
         geom.index_origin[axis];
}

namespace {

double wrap_coord(double x, double lo, double hi) {
  const double period = hi - lo;
  if (!(period > 0.0)) {
    return x;
  }
  double y = std::fmod(x - lo, period);
  if (y < 0.0) {
    y += period;
  }
  return lo + y;
}

} // namespace

bool solve_3x3(double a[3][3], double b[3], double x[3]) {
  double m[3][4] = {
      {a[0][0], a[0][1], a[0][2], b[0]},
      {a[1][0], a[1][1], a[1][2], b[1]},
      {a[2][0], a[2][1], a[2][2], b[2]},
  };

  for (int col = 0; col < 3; ++col) {
    int piv = col;
    double best = std::abs(m[col][col]);
    for (int r = col + 1; r < 3; ++r) {
      const double cand = std::abs(m[r][col]);
      if (cand > best) {
        best = cand;
        piv = r;
      }
    }
    if (best < 1e-30) {
      return false;
    }
    if (piv != col) {
      for (int c = col; c < 4; ++c) {
        std::swap(m[col][c], m[piv][c]);
      }
    }
    const double inv = 1.0 / m[col][col];
    for (int c = col; c < 4; ++c) {
      m[col][c] *= inv;
    }
    for (int r = 0; r < 3; ++r) {
      if (r == col) {
        continue;
      }
      const double f = m[r][col];
      if (f == 0.0) {
        continue;
      }
      for (int c = col; c < 4; ++c) {
        m[r][c] -= f * m[col][c];
      }
    }
  }

  x[0] = m[0][3];
  x[1] = m[1][3];
  x[2] = m[2][3];
  return true;
}

namespace {

std::optional<double> patch_value_at(const SamplePatch &p, int32_t i, int32_t j,
                                     int32_t k) {
  if (i < p.box.lo[0] || i > p.box.hi[0] || j < p.box.lo[1] ||
      j > p.box.hi[1] || k < p.box.lo[2] || k > p.box.hi[2]) {
    return std::nullopt;
  }
  const int32_t li = i - p.box.lo[0];
  const int32_t lj = j - p.box.lo[1];
  const int32_t lk = k - p.box.lo[2];
  return p.values(li, lj, lk);
}

} // namespace

std::optional<SamplePoint>
composite_sample_at(std::span<const SamplePatch> patches, int finest_level,
                    const int32_t domain_lo[3], const int32_t domain_hi[3],
                    const bool is_periodic[3], const double domain_lo_edge[3],
                    const double domain_hi_edge[3], double x, double y,
                    double z) {
  double xyz[3] = {x, y, z};
  for (int ax = 0; ax < 3; ++ax) {
    if (is_periodic[ax]) {
      xyz[ax] = wrap_coord(xyz[ax], domain_lo_edge[ax], domain_hi_edge[ax]);
    } else if (xyz[ax] < domain_lo_edge[ax] || xyz[ax] >= domain_hi_edge[ax]) {
      return std::nullopt;
    }
  }

  for (int lev = finest_level; lev >= 0; --lev) {
    for (const auto &p : patches) {
      if (p.level != lev) {
        continue;
      }
      int32_t idx[3];
      for (int ax = 0; ax < 3; ++ax) {
        idx[ax] = coord_to_index(p.geom, ax, xyz[ax]);
      }
      if (idx[0] < p.box.lo[0] || idx[0] > p.box.hi[0] ||
          idx[1] < p.box.lo[1] || idx[1] > p.box.hi[1] ||
          idx[2] < p.box.lo[2] || idx[2] > p.box.hi[2]) {
        continue;
      }
      auto value = patch_value_at(p, idx[0], idx[1], idx[2]);
      if (!value.has_value()) {
        continue;
      }
      SamplePoint out;
      out.x = cell_center(p.geom, 0, idx[0]);
      out.y = cell_center(p.geom, 1, idx[1]);
      out.z = cell_center(p.geom, 2, idx[2]);
      out.value = *value;
      return out;
    }
  }
  return std::nullopt;
}

} // namespace kangaroo
