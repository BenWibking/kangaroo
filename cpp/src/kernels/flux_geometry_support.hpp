#pragma once

#include <cstdint>

namespace kangaroo {

struct CellIndexRange {
  int first = 0;
  int last = -1;

  bool empty() const { return first > last; }
};

double min_dist_sq_to_interval(double a0, double a1);
double max_dist_sq_to_interval(double a0, double a1);
bool sphere_may_intersect_cell(double radius2, double x0, double x1, double y0,
                               double y1, double z0, double z1);
bool cylinder_may_intersect_cell(double radius2, double height, double x0,
                                 double x1, double y0, double y1, double z0,
                                 double z1);
CellIndexRange cells_intersecting_axis_band(double band_lo, double band_hi,
                                            double x0, double dx,
                                            int32_t index_origin,
                                            int32_t block_lo, int32_t block_hi);
double plane_box_section_area(double x0, double x1, double y0, double y1,
                              double z0, double z1, double nx, double ny,
                              double nz, double d);
double spherical_section_area_in_intersecting_cell(double radius, double x0,
                                                   double x1, double y0,
                                                   double y1, double z0,
                                                   double z1);
double cylindrical_section_area_in_intersecting_cell(double radius,
                                                     double height, double x0,
                                                     double x1, double y0,
                                                     double y1, double z0,
                                                     double z1);

} // namespace kangaroo
