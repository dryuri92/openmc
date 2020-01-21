#include "openmc/mesh.h"

#include <algorithm> // for copy, equal, min, min_element
#include <cstddef> // for size_t
#include <cmath>  // for ceil
#include <string>

#ifdef OPENMC_MPI
#include "mpi.h"
#endif
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/message_passing.h"
#include "openmc/search.h"
#include "openmc/tallies/filter.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace model {

std::vector<std::unique_ptr<RegularMesh>> meshes;
std::unordered_map<int32_t, int32_t> mesh_map;

} // namespace model

//==============================================================================
// Helper functions
//==============================================================================

//! Update an intersection point if the given candidate is closer.
//
//! The first 6 arguments are coordinates for the starting point of a particle
//! and its intersection with a mesh surface.  If the distance between these
//! two points is shorter than the given `min_distance`, then the `r` argument
//! will be updated to match the intersection point, and `min_distance` will
//! also be updated.

inline bool check_intersection_point(double x1, double x0, double y1,
  double y0, double z1, double z0, Position& r, double& min_distance)
{
  double dist = std::pow(x1-x0, 2) + std::pow(y1-y0, 2) + std::pow(z1-z0, 2);
  if (dist < min_distance) {
    r.x = x1;
    r.y = y1;
    r.z = z1;
    min_distance = dist;
    return true;
  }
  return false;
}

//==============================================================================
// Mesh implementation
//==============================================================================
RegularMesh::RegularMesh(){

}

RegularMesh::RegularMesh(pugi::xml_node node)

{
	// Copy mesh id
	  if (check_for_node(node, "id")) {
	    id_ = std::stoi(get_node_value(node, "id"));
        volume_frac_=0.0;
        n_dimension_ = 0;
	    // Check to make sure 'id' hasn't been used
	    if (model::mesh_map.find(id_) != model::mesh_map.end()) {
	      fatal_error("Two or more meshes use the same unique ID: " +
	        std::to_string(id_));
	    }
	  }

}

//==============================================================================

//==============================================================================
// RegularMesh implementation
//==============================================================================

RectMesh::RectMesh()
: RegularMesh {}
{

}
RectMesh::RectMesh(pugi::xml_node node)
: RegularMesh {node}
{


  type_ = MeshType::rect;
  // Read mesh type
  if (check_for_node(node, "type")) {
    auto temp = get_node_value(node, "type", true, true);
    if (temp == "regular") {
      // TODO: move elsewhere
    	type_ = MeshType::rect;
    } else {
      fatal_error("Invalid mesh type: " + temp);
    }
  }

  // Determine number of dimensions for mesh
  if (check_for_node(node, "dimension")) {
    shape_ = get_node_xarray<int>(node, "dimension");
    int n = n_dimension_ = shape_.size();
    if (n != 1 && n != 2 && n != 3) {
      fatal_error("Mesh must be one, two, or three dimensions.");
    }

    // Check that dimensions are all greater than zero
    if (xt::any(shape_ <= 0)) {
      fatal_error("All entries on the <dimension> element for a tally "
        "mesh must be positive.");
    }
  }

  // Check for lower-left coordinates
  if (check_for_node(node, "lower_left")) {
    // Read mesh lower-left corner location
    lower_left_ = get_node_xarray<double>(node, "lower_left");
  } else {
    fatal_error("Must specify <lower_left> on a mesh.");
  }

  if (check_for_node(node, "width")) {
    // Make sure both upper-right or width were specified
    if (check_for_node(node, "upper_right")) {
      fatal_error("Cannot specify both <upper_right> and <width> on a mesh.");
    }

    width_ = get_node_xarray<double>(node, "width");

    // Check to ensure width has same dimensions
    auto n = width_.size();
    if (n != lower_left_.size()) {
      fatal_error("Number of entries on <width> must be the same as "
        "the number of entries on <lower_left>.");
    }

    // Check for negative widths
    if (xt::any(width_ < 0.0)) {
      fatal_error("Cannot have a negative <width> on a tally mesh.");
    }

    // Set width and upper right coordinate
    upper_right_ = xt::eval(lower_left_ + shape_ * width_);

  } else if (check_for_node(node, "upper_right")) {
    upper_right_ = get_node_xarray<double>(node, "upper_right");

    // Check to ensure width has same dimensions
    auto n = upper_right_.size();
    if (n != lower_left_.size()) {
      fatal_error("Number of entries on <upper_right> must be the "
        "same as the number of entries on <lower_left>.");
    }

    // Check that upper-right is above lower-left
    if (xt::any(upper_right_ < lower_left_)) {
      fatal_error("The <upper_right> coordinates must be greater than "
        "the <lower_left> coordinates on a tally mesh.");
    }

    // Set width and upper right coordinate
    width_ = xt::eval((upper_right_ - lower_left_) / shape_);
  } else {
    fatal_error("Must specify either <upper_right> and <width> on a mesh.");
  }

  if (shape_.dimension() > 0) {
    if (shape_.size() != lower_left_.size()) {
      fatal_error("Number of entries on <lower_left> must be the same "
        "as the number of entries on <dimension>.");
    }

    // Set volume fraction
    volume_frac_ = 1.0/xt::prod(shape_)();
  }
}

int RectMesh::get_bin(Position r) const
{
  // Loop over the dimensions of the mesh
  for (int i = 0; i < n_dimension_; ++i) {
    // Check for cases where particle is outside of mesh
    if (r[i] < lower_left_[i]) {
      return -1;
    } else if (r[i] > upper_right_[i]) {
      return -1;
    }
  }

  // Determine indices
  int ijk[n_dimension_];
  bool in_mesh;
  get_indices(r, ijk, &in_mesh);
  if (!in_mesh) return -1;

  // Convert indices to bin
  return get_bin_from_indices(ijk);
}

int RectMesh::get_bin_from_indices(const int* ijk) const
{
  switch (n_dimension_) {
    case 1:
      return ijk[0] - 1;
    case 2:
      return (ijk[1] - 1)*shape_[0] + ijk[0] - 1;
    case 3:
      return ((ijk[2] - 1)*shape_[1] + (ijk[1] - 1))*shape_[0] + ijk[0] - 1;
    default:
      throw std::runtime_error{"Invalid number of mesh dimensions"};
  }
}

void RectMesh::get_indices(Position r, int* ijk, bool* in_mesh) const
{
  // Find particle in mesh
  *in_mesh = true;
  for (int i = 0; i < n_dimension_; ++i) {
    ijk[i] = std::ceil((r[i] - lower_left_[i]) / width_[i]);

    // Check if indices are within bounds
    if (ijk[i] < 1 || ijk[i] > shape_[i]) *in_mesh = false;
  }
}

void RectMesh::get_indices_from_bin(int bin, int* ijk) const
{
  if (n_dimension_ == 1) {
    ijk[0] = bin + 1;
  } else if (n_dimension_ == 2) {
    ijk[0] = bin % shape_[0] + 1;
    ijk[1] = bin / shape_[0] + 1;
  } else if (n_dimension_ == 3) {
    ijk[0] = bin % shape_[0] + 1;
    ijk[1] = (bin % (shape_[0] * shape_[1])) / shape_[0] + 1;
    ijk[2] = bin / (shape_[0] * shape_[1]) + 1;
  }
}

bool RectMesh::intersects(Position r0, Position r1) const
{
  switch(n_dimension_) {
    case 1:
      return intersects_1d(r0, r1);
    case 2:
      return intersects_2d(r0, r1);
    case 3:
      return intersects_3d(r0, r1);
    default:
      throw std::runtime_error{"Invalid number of mesh dimensions."};
  }
}

bool RectMesh::intersects_1d(Position r0, Position r1) const
{
  // Copy coordinates of mesh lower_left and upper_right
  double left = lower_left_[0];
  double right = upper_right_[0];

  // Check if line intersects either left or right surface
  if (r0.x < left) {
    return r1.x > left;
  } else if (r0.x < right) {
    return r1.x < left || r1.x > right;
  } else {
    return r1.x < right;
  }
}

bool RectMesh::intersects_2d(Position r0, Position r1) const
{
  // Copy coordinates of starting point
  double x0 = r0.x;
  double y0 = r0.y;

  // Copy coordinates of ending point
  double x1 = r1.x;
  double y1 = r1.y;

  // Copy coordinates of mesh lower_left
  double xm0 = lower_left_[0];
  double ym0 = lower_left_[1];

  // Copy coordinates of mesh upper_right
  double xm1 = upper_right_[0];
  double ym1 = upper_right_[1];

  // Check if line intersects left surface -- calculate the intersection point y
  if ((x0 < xm0 && x1 > xm0) || (x0 > xm0 && x1 < xm0)) {
    double yi = y0 + (xm0 - x0) * (y1 - y0) / (x1 - x0);
    if (yi >= ym0 && yi < ym1) {
      return true;
    }
  }

  // Check if line intersects back surface -- calculate the intersection point
  // x
  if ((y0 < ym0 && y1 > ym0) || (y0 > ym0 && y1 < ym0)) {
    double xi = x0 + (ym0 - y0) * (x1 - x0) / (y1 - y0);
    if (xi >= xm0 && xi < xm1) {
      return true;
    }
  }

  // Check if line intersects right surface -- calculate the intersection
  // point y
  if ((x0 < xm1 && x1 > xm1) || (x0 > xm1 && x1 < xm1)) {
    double yi = y0 + (xm1 - x0) * (y1 - y0) / (x1 - x0);
    if (yi >= ym0 && yi < ym1) {
      return true;
    }
  }

  // Check if line intersects front surface -- calculate the intersection point
  // x
  if ((y0 < ym1 && y1 > ym1) || (y0 > ym1 && y1 < ym1)) {
    double xi = x0 + (ym1 - y0) * (x1 - x0) / (y1 - y0);
    if (xi >= xm0 && xi < xm1) {
      return true;
    }
  }
  return false;
}

bool RectMesh::intersects_3d(Position r0, Position r1) const
{
  // Copy coordinates of starting point
  double x0 = r0.x;
  double y0 = r0.y;
  double z0 = r0.z;

  // Copy coordinates of ending point
  double x1 = r1.x;
  double y1 = r1.y;
  double z1 = r1.z;

  // Copy coordinates of mesh lower_left
  double xm0 = lower_left_[0];
  double ym0 = lower_left_[1];
  double zm0 = lower_left_[2];

  // Copy coordinates of mesh upper_right
  double xm1 = upper_right_[0];
  double ym1 = upper_right_[1];
  double zm1 = upper_right_[2];

  // Check if line intersects left surface -- calculate the intersection point
  // (y,z)
  if ((x0 < xm0 && x1 > xm0) || (x0 > xm0 && x1 < xm0)) {
    double yi = y0 + (xm0 - x0) * (y1 - y0) / (x1 - x0);
    double zi = z0 + (xm0 - x0) * (z1 - z0) / (x1 - x0);
    if (yi >= ym0 && yi < ym1 && zi >= zm0 && zi < zm1) {
      return true;
    }
  }

  // Check if line intersects back surface -- calculate the intersection point
  // (x,z)
  if ((y0 < ym0 && y1 > ym0) || (y0 > ym0 && y1 < ym0)) {
    double xi = x0 + (ym0 - y0) * (x1 - x0) / (y1 - y0);
    double zi = z0 + (ym0 - y0) * (z1 - z0) / (y1 - y0);
    if (xi >= xm0 && xi < xm1 && zi >= zm0 && zi < zm1) {
      return true;
    }
  }

  // Check if line intersects bottom surface -- calculate the intersection
  // point (x,y)
  if ((z0 < zm0 && z1 > zm0) || (z0 > zm0 && z1 < zm0)) {
    double xi = x0 + (zm0 - z0) * (x1 - x0) / (z1 - z0);
    double yi = y0 + (zm0 - z0) * (y1 - y0) / (z1 - z0);
    if (xi >= xm0 && xi < xm1 && yi >= ym0 && yi < ym1) {
      return true;
    }
  }

  // Check if line intersects right surface -- calculate the intersection point
  // (y,z)
  if ((x0 < xm1 && x1 > xm1) || (x0 > xm1 && x1 < xm1)) {
    double yi = y0 + (xm1 - x0) * (y1 - y0) / (x1 - x0);
    double zi = z0 + (xm1 - x0) * (z1 - z0) / (x1 - x0);
    if (yi >= ym0 && yi < ym1 && zi >= zm0 && zi < zm1) {
      return true;
    }
  }

  // Check if line intersects front surface -- calculate the intersection point
  // (x,z)
  if ((y0 < ym1 && y1 > ym1) || (y0 > ym1 && y1 < ym1)) {
    double xi = x0 + (ym1 - y0) * (x1 - x0) / (y1 - y0);
    double zi = z0 + (ym1 - y0) * (z1 - z0) / (y1 - y0);
    if (xi >= xm0 && xi < xm1 && zi >= zm0 && zi < zm1) {
      return true;
    }
  }

  // Check if line intersects top surface -- calculate the intersection point
  // (x,y)
  if ((z0 < zm1 && z1 > zm1) || (z0 > zm1 && z1 < zm1)) {
    double xi = x0 + (zm1 - z0) * (x1 - x0) / (z1 - z0);
    double yi = y0 + (zm1 - z0) * (y1 - y0) / (z1 - z0);
    if (xi >= xm0 && xi < xm1 && yi >= ym0 && yi < ym1) {
      return true;
    }
  }
  return false;
}

void RectMesh::bins_crossed(const Particle* p, std::vector<int>& bins,
                               std::vector<double>& lengths) const
{
  constexpr int MAX_SEARCH_ITER = 100;

  // ========================================================================
  // Determine if the track intersects the tally mesh.

  // Copy the starting and ending coordinates of the particle.  Offset these
  // just a bit for the purposes of determining if there was an intersection
  // in case the mesh surfaces coincide with lattice/geometric surfaces which
  // might produce finite-precision errors.
  Position last_r {p->r_last_};
  Position r {p->r()};
  Direction u {p->u()};

  Position r0 = last_r + TINY_BIT*u;
  Position r1 = r - TINY_BIT*u;

  // Determine indices for starting and ending location.
  int n = n_dimension_;
  int ijk0[n], ijk1[n];
  bool start_in_mesh;
  get_indices(r0, ijk0, &start_in_mesh);
  bool end_in_mesh;
  get_indices(r1, ijk1, &end_in_mesh);

  // Check if the track intersects any part of the mesh.
  if (!start_in_mesh && !end_in_mesh) {
    if (!intersects(r0, r1)) return;
  }

  // ========================================================================
  // Figure out which mesh cell to tally.

  // Copy the un-modified coordinates the particle direction.
  r0 = last_r;
  r1 = r;

  // Compute the length of the entire track.
  double total_distance = (r1 - r0).norm();

  // We are looking for the first valid mesh bin.  Check to see if the
  // particle starts inside the mesh.
  if (!start_in_mesh) {
    double d[n];

    // The particle does not start in the mesh.  Note that we nudged the
    // start and end coordinates by a TINY_BIT each so we will have
    // difficulty resolving tracks that are less than 2*TINY_BIT in length.
    // If the track is that short, it is also insignificant so we can
    // safely ignore it in the tallies.
    if (total_distance < 2*TINY_BIT) return;

    // The particle does not start in the mesh so keep iterating the ijk0
    // indices to cross the nearest mesh surface until we've found a valid
    // bin.  MAX_SEARCH_ITER prevents an infinite loop.
    int search_iter = 0;
    int j;
    bool in_mesh = true;
    for (int i = 0; i < n; ++i) {
      if (ijk0[i] < 1 || ijk0[i] > shape_[i]) {
       in_mesh = false;
       break;
      }
    }
    while (!in_mesh) {
      if (search_iter == MAX_SEARCH_ITER) {
        warning("Failed to find a mesh intersection on a tally mesh filter.");
        return;
      }

      for (j = 0; j < n; ++j) {
        if (std::fabs(u[j]) < FP_PRECISION) {
          d[j] = INFTY;
        } else if (u[j] > 0.0) {
          double xyz_cross = lower_left_[j] + ijk0[j] * width_[j];
          d[j] = (xyz_cross - r0[j]) / u[j];
        } else {
          double xyz_cross = lower_left_[j] + (ijk0[j] - 1) * width_[j];
          d[j] = (xyz_cross - r0[j]) / u[j];
        }
      }

      j = std::min_element(d, d+n) - d;// Very tricky method to get an index of minimal element array
      if (u[j] > 0.0) {
        ++ijk0[j];
      } else {
        --ijk0[j];
      }

      ++search_iter;
      in_mesh = true;
      for (int i = 0; i < n; ++i) {
        if (ijk0[i] < 1 || ijk0[i] > shape_[i]) {
         in_mesh = false;
         break;
        }
      }
    }

    // Advance position
    r0 += d[j] * u;
  }

  while (true) {
    // ========================================================================
    // Compute the length of the track segment in the each mesh cell and return

    if (std::equal(ijk0, ijk0+n, ijk1)) {
      // The track ends in this cell.  Use the particle end location rather
      // than the mesh surface.
      double distance = (r1 - r0).norm();
      bins.push_back(get_bin_from_indices(ijk0));
      lengths.push_back(distance / total_distance);
      break;
    }

    // The track exits this cell.  Determine the distance to the closest mesh
    // surface.
    double d[n];
    for (int k = 0; k < n; ++k) {
      if (std::fabs(u[k]) < FP_PRECISION) {
        d[k] = INFTY;
      } else if (u[k] > 0) {
        double xyz_cross = lower_left_[k] + ijk0[k] * width_[k];
        d[k] = (xyz_cross - r0[k]) / u[k];
      } else {
        double xyz_cross = lower_left_[k] + (ijk0[k] - 1) * width_[k];
        d[k] = (xyz_cross - r0[k]) / u[k];
      }
    }

    // Assign the next tally bin and the score.
    auto j = std::min_element(d, d+n) - d;
    double distance = d[j];
    bins.push_back(get_bin_from_indices(ijk0));
    lengths.push_back(distance / total_distance);

    // Translate the starting coordintes by the distance to the oncoming mesh
    // surface.
    r0 += distance * u;

    // Increment the indices into the next mesh cell.
    if (u[j] > 0.0) {
      ++ijk0[j];
    } else {
      --ijk0[j];
    }

    // If the next indices are invalid, then the track has left the mesh and
    // we are done.
    bool in_mesh = true;
    for (int i = 0; i < n; ++i) {
      if (ijk0[i] < 1 || ijk0[i] > shape_[i]) {
        in_mesh = false;
        break;
      }
    }
    if (!in_mesh) break;
  }
}

void RectMesh::surface_bins_crossed(const Particle* p, std::vector<int>& bins) const
{
  // ========================================================================
  // Determine if the track intersects the tally mesh.

  // Copy the starting and ending coordinates of the particle.
  Position r0 {p->r_last_current_};
  Position r1 {p->r()};
  Direction u {p->u()};

  // Determine indices for starting and ending location.
  int n = n_dimension_;
  int ijk0[n], ijk1[n];
  bool start_in_mesh;
  get_indices(r0, ijk0, &start_in_mesh);
  bool end_in_mesh;
  get_indices(r1, ijk1, &end_in_mesh);

  // Check if the track intersects any part of the mesh.
  if (!start_in_mesh && !end_in_mesh) {
    if (!intersects(r0, r1)) return;
  }

  // ========================================================================
  // Figure out which mesh cell to tally.

  // Calculate number of surface crossings
  int n_cross = 0;
  for (int i = 0; i < n; ++i) n_cross += std::abs(ijk1[i] - ijk0[i]);
  if (n_cross == 0) return;

  // Bounding coordinates
  Position xyz_cross;
  for (int i = 0; i < n; ++i) {
    if (u[i] > 0.0) {
      xyz_cross[i] = lower_left_[i] + ijk0[i] * width_[i];
    } else {
      xyz_cross[i] = lower_left_[i] + (ijk0[i] - 1) * width_[i];
    }
  }

  for (int j = 0; j < n_cross; ++j) {
    // Set the distances to infinity
    Position d {INFTY, INFTY, INFTY};

    // Determine closest bounding surface. We need to treat
    // special case where the cosine of the angle is zero since this would
    // result in a divide-by-zero.
    double distance = INFTY;
    for (int i = 0; i < n; ++i) {
      if (u[i] == 0) {
        d[i] = INFINITY;
      } else {
        d[i] = (xyz_cross[i] - r0[i])/u[i];
      }
      distance = std::min(distance, d[i]);
    }

    // Loop over the dimensions
    for (int i = 0; i < n; ++i) {
      // Check whether distance is the shortest distance
      if (distance == d[i]) {

        // Check whether the current indices are within the mesh bounds
        bool in_mesh = true;
        for (int j = 0; j < n; ++j) {
          if (ijk0[j] < 1 || ijk0[j] > shape_[j]) {
            in_mesh = false;
            break;
          }
        }

        // Check whether particle is moving in positive i direction
        if (u[i] > 0) {

          // Outward current on i max surface
          if (in_mesh) {
            int i_surf = 4*i + 3;
            int i_mesh = get_bin_from_indices(ijk0);
            int i_bin = 4*n*i_mesh + i_surf - 1;

            bins.push_back(i_bin);
          }

          // Advance position
          ++ijk0[i];
          xyz_cross[i] += width_[i];
          in_mesh = true;
          for (int j = 0; j < n; ++j) {
            if (ijk0[j] < 1 || ijk0[j] > shape_[j]) {
              in_mesh = false;
              break;
            }
          }

          // If the particle crossed the surface, tally the inward current on
          // i min surface
          if (in_mesh) {
            int i_surf = 4*i + 2;
            int i_mesh = get_bin_from_indices(ijk0);
            int i_bin = 4*n*i_mesh + i_surf - 1;

            bins.push_back(i_bin);
          }

        } else {
          // The particle is moving in the negative i direction

          // Outward current on i min surface
          if (in_mesh) {
            int i_surf = 4*i + 1;
            int i_mesh = get_bin_from_indices(ijk0);
            int i_bin = 4*n*i_mesh + i_surf - 1;

            bins.push_back(i_bin);
          }

          // Advance position
          --ijk0[i];
          xyz_cross[i] -= width_[i];
          in_mesh = true;
          for (int j = 0; j < n; ++j) {
            if (ijk0[j] < 1 || ijk0[j] > shape_[j]) {
              in_mesh = false;
              break;
            }
          }

          // If the particle crossed the surface, tally the inward current on
          // i max surface
          if (in_mesh) {
            int i_surf = 4*i + 4;
            int i_mesh = get_bin_from_indices(ijk0);
            int i_bin = 4*n*i_mesh + i_surf - 1;

            bins.push_back(i_bin);
          }
        }
      }
    }

    // Calculate new coordinates
    r0 += distance * u;
  }
}

void RectMesh::to_hdf5(hid_t group) const
{
  hid_t mesh_group = create_group(group, "mesh " + std::to_string(id_));

  write_dataset(mesh_group, "type", "regular");
  write_dataset(mesh_group, "dimension", shape_);
  write_dataset(mesh_group, "lower_left", lower_left_);
  write_dataset(mesh_group, "upper_right", upper_right_);
  write_dataset(mesh_group, "width", width_);

  close_group(mesh_group);
}

xt::xarray<double>
RectMesh::count_sites(const std::vector<Particle::Bank>& bank,
  bool* outside) const
{
  // Determine shape of array for counts
  std::size_t m = xt::prod(shape_)();
  std::vector<std::size_t> shape = {m};

  // Create array of zeros
  xt::xarray<double> cnt {shape, 0.0};
  bool outside_ = false;

  for (const auto& site : bank) {
    // determine scoring bin for entropy mesh
    int mesh_bin = get_bin(site.r);

    // if outside mesh, skip particle
    if (mesh_bin < 0) {
      outside_ = true;
      continue;
    }

    // Add to appropriate bin
    cnt(mesh_bin) += site.wgt;
  }

  // Create copy of count data
  int total = cnt.size();
  double* cnt_reduced = new double[total];

#ifdef OPENMC_MPI
  // collect values from all processors
  MPI_Reduce(cnt.data(), cnt_reduced, total, MPI_DOUBLE, MPI_SUM, 0,
    mpi::intracomm);

  // Check if there were sites outside the mesh for any processor
  if (outside) {
    MPI_Reduce(&outside_, outside, 1, MPI_C_BOOL, MPI_LOR, 0, mpi::intracomm);
  }
#else
  std::copy(cnt.data(), cnt.data() + total, cnt_reduced);
  if (outside) *outside = outside_;
#endif

  // Adapt reduced values in array back into an xarray
  auto arr = xt::adapt(cnt_reduced, total, xt::acquire_ownership(), shape);
  xt::xarray<double> counts = arr;

  return counts;
}
//
//==============================================================================
//! Tessellation of n-dimensional Euclidean space by hexagonal prism
//==============================================================================
HexMesh::HexMesh(pugi::xml_node node)
: RegularMesh {node}
{

	write_message("Reading info hex!!!!!!!");
	type_ = MeshType::hex;
  // Read mesh type
  if (check_for_node(node, "type")) {
    auto temp = get_node_value(node, "type", true, true);

  }

  // Read the number of lattice cells in each dimension.
   n_rings_ = std::stoi(get_node_value(node, "n_rings"));
   if (check_for_node(node, "n_axial")) {
   n_axial_ = std::stoi(get_node_value(node, "n_axial"));
   }
   //Shapes of a hexagonal mesh

  // Determine number of dimensions for mesh
    if (n_axial_>1){
    	n_dimension_ = 3;
    }


      shape_ = xt::ones<int>({3});
      shape_(0) = 2*n_rings_- 1;//3*(n_rings_- 1)*n_rings_ + 1;
      shape_(1) =  2*n_rings_- 1;
      shape_(2) = n_axial_;

    int n = n_dimension_;
    if (n != 1 && n != 2 && n != 3) {
      fatal_error("Mesh must be one, two, or three dimensions.");
    }

    // Check that dimensions are all greater than zero
    if (xt::any(shape_ <= 0)) {
      fatal_error("All entries on the <dimension> element for a tally "
        "mesh must be positive.");
    }


  // Check for lower-left coordinates
  if (check_for_node(node, "center")) {
    // Read mesh lower-left corner location
	  xt::xarray<double> loccenter { get_node_xarray<double>(node, "center")};
     center_.x = loccenter(0);
     center_.y = loccenter(1);
     if (n_dimension_ > 2) {
    	 center_.z = loccenter(2);
     }
  } else {
    fatal_error("Must specify <center> on a mesh.");
  }

  if (check_for_node(node, "width")) {
    // Make sure  width were specified

    width_ = get_node_xarray<double>(node, "width");

    // Check to ensure width has same dimensions
    auto n = width_.size();

  }

  if (n_axial_ > 1) {
	  if (check_for_node(node, "discretez")) {
		  discretez_ = get_node_xarray<double>(node, "discretez");
	  }
	  else {
	      fatal_error("Must specify discretez on a hexagonal mesh.");}


  int iz {0};
  heightz_ = xt::zeros<double>({discretez_.size() + 1});
  lower_left_ = xt::zeros<double>({n_dimension_});
  upper_right_ = xt::zeros<double>({n_dimension_});
  heightz_(0) = center_.z;
  for (auto it = discretez_.begin();it!=discretez_.end();it++ )
  {
	  heightz_(iz+1) = heightz_(iz) + *it;
	  iz++;
  }
  std :: cout << "CHECK AXIAL SIZE";
  for (auto it = heightz_.begin();it!=heightz_.end();it++ )
    {
  	  std::cout << *it << std :: endl;

    }


  }

  // Check for negative widths
   if (xt::any(width_ < 0.0)) {
     fatal_error("Cannot have a negative <width> on a tally mesh.");
   }
  if (shape_.dimension() > 0) {


    // Set volume fraction
    volume_frac_ = 1.0/xt::prod(shape_)();
  }
  _hpp = 2*(n_rings_ + (n_rings_ - 1)/2.0)*width_(0)/sqrt(3.0);

  float sqt3 {sqrt(3.0)/2.0};
  lines = xt::xarray<Position>(_lineshapes);
  std::cout << "WRITE-LINES " <<n_rings_ << " " << width_(0) ;
  // right - up boundary
  lines(0,0).x = _hpp/(sqt3*2.0);
  lines(0,0).y = _hpp/2.0;
  lines(0,1).x = _hpp/sqt3;
  lines(0,1).y = 0.0;
  // right - down boundary
  lines(1,0).x = _hpp/sqt3;
  lines(1,0).y = 0.0;
  lines(1,1).x = _hpp/(sqt3*2.0);
  lines(1,1).y = -_hpp/2.;
  // down boundary
  lines(2,0).x = _hpp/(sqt3*2.0);
  lines(2,0).y = -_hpp/2.;
  lines(2,1).x = -_hpp/(sqt3*2.0);
  lines(2,1).y = -_hpp/2.;
  // left - down boundary
  lines(3,0).x = -_hpp/(sqt3*2.0);
  lines(3,0).y = -_hpp/2.;
  lines(3,1).x = -_hpp/sqt3;
  lines(3,1).y = 0.0;
  // left - up boundary
  lines(4,0).x = -_hpp/sqt3;
  lines(4,0).y = 0.0;
  lines(4,1).x = -_hpp/(sqt3*2.0);
  lines(4,1).y = _hpp/2.0;
  // up  boundary
  lines(5,0).x = -_hpp/(sqt3*2.0);
  lines(5,0).y = _hpp/2.0;
  lines(5,1).x = _hpp/(sqt3*2.0);
  lines(5,1).y = _hpp/2.0;
  //std::cout << "Y:" << lines(5,1).y << " X: " << lines(4,0).x;
}
//=================================================================
 void HexMesh::bins_crossed(const Particle* p, std::vector<int>& bins,
                               std::vector<double>& lengths) const
{
  constexpr int MAX_SEARCH_ITER = 100;

  // ========================================================================
  // Determine if the track intersects the tally mesh.

  // Copy the starting and ending coordinates of the particle.  Offset these
  // just a bit for the purposes of determining if there was an intersection
  // in case the mesh surfaces coincide with lattice/geometric surfaces which
  // might produce finite-precision errors.
  int iter {0};
  Position last_r {p->r_last_};
  Position r {p->r()};
  Direction u {p->u()};

  Position r0 = last_r + TINY_BIT*u;
  Position r1 = r - TINY_BIT*u;

  // Determine indices for starting and ending location.
  int n = n_dimension_;
  int ijk0[n], ijk1[n];
  bool start_in_mesh;
  bool finallycross;
  bool end_in_mesh;
  crossing(r0,r1,&ijk0[0],&ijk1[0],finallycross,start_in_mesh,end_in_mesh);

  double this_d;
  // Check if the track intersects any part of the mesh.
  if (!start_in_mesh && !end_in_mesh) {
    if (!finallycross) return;
  }

  // ========================================================================
  // Figure out which mesh cell to tally.

  // Copy the un-modified coordinates the particle direction.
  r0 = last_r;
  r1 = r;

  // Compute the length of the entire track.
  double total_distance = (r1 - r0).norm();
  // The particle does not start in the mesh.  Note that we nudged the
     // start and end coordinates by a TINY_BIT each so we will have
     // difficulty resolving tracks that are less than 2*TINY_BIT in length.
     // If the track is that short, it is also insignificant so we can
     // safely ignore it in the tallies.
   if (total_distance < 2*TINY_BIT) return;
  //
  // Compute the direction on the hexagonal basis.
   double beta_dir = u.x*0.5  + u.y * std::sqrt(3.0) / 2.0;
   double gamma_dir = u.x*0.5  - u.y * std::sqrt(3.0) / 2.0;

    // Note that hexagonal lattice distance calculations are performed
    // using the particle's coordinates relative to the neighbor lattice
    // cells, not relative to the particle's current cell.  This is done
    // because there is significant disagreement between neighboring cells
    // on where the lattice boundary is due to finite precision issues.

   while (true) {
    // ========================================================================
    // Compute the length of the track segment in the each mesh cell and return

	iter++;
    if (std::equal(ijk0, ijk0+n, ijk1)) {
      // The track ends in this cell.  Use the particle end location rather
      // than the mesh surface.
     if (end_in_mesh){

      double distance = (r1 - r0).norm();
      bins.push_back(get_bin_from_indices(ijk0));
      lengths.push_back(distance / total_distance);

     }
      break;
    }

    // The track exits this cell.  Determine the distance to the closest mesh
    // surface.
    // Upper-right and lower-left sides.
        double d {INFTY};
        std::array<int, 3> lattice_trans;
        double edge = -copysign(0.5*width_[0], beta_dir);  // Oncoming edge
        Position r_t;
        if (beta_dir > 0) {
          const std::array<int, 3> i_xyz_t {ijk0[0], ijk0[1]+1, ijk0[2]};
          r_t = get_local_position(r0, i_xyz_t);
        } else {
          const std::array<int, 3> i_xyz_t {ijk0[0], ijk0[1]-1, ijk0[2]};
          r_t = get_local_position(r0, i_xyz_t);
        }
        double beta = r_t.x / 2.0 + r_t.y * std::sqrt(3.0) / 2.0;
        if ((std::abs(beta - edge) > FP_PRECISION) && beta_dir != 0) {
          d = (edge - beta) / beta_dir;
          if (beta_dir > 0) {
            lattice_trans = {0, 1, 0};
          } else {
            lattice_trans = {0, -1, 0};
          }
        }

        // Lower-right and upper-left sides.
        edge = -copysign(0.5*width_[0], gamma_dir);
        if (gamma_dir > 0) {
            const std::array<int, 3> i_xyz_t {ijk0[0]+1, ijk0[1]-1, ijk0[2]};
            r_t = get_local_position(r0, i_xyz_t);
          } else {
            const std::array<int, 3> i_xyz_t {ijk0[0]-1, ijk0[1]+1, ijk0[2]};
            r_t = get_local_position(r0, i_xyz_t);
          }
          double gamma = r_t.x / 2.0 - r_t.y * std::sqrt(3.0) / 2.0;
          if ((std::abs(gamma - edge) > FP_PRECISION) && gamma_dir != 0) {
            this_d = (edge - gamma) / gamma_dir;
            if (this_d < d) {
              if (gamma_dir > 0) {
                lattice_trans = {1, -1, 0};
              } else {
                lattice_trans = {-1, 1, 0};
              }
              d = this_d;
            }
          }

        // Upper and lower sides.
          edge = -copysign(0.5*width_[0], u.x);
            if (u.x > 0) {
              const std::array<int, 3> i_xyz_t {ijk0[0]+1, ijk0[1], ijk0[2]};
              r_t = get_local_position(r0, i_xyz_t);
            } else {
              const std::array<int, 3> i_xyz_t {ijk0[0]-1, ijk0[1], ijk0[2]};
              r_t = get_local_position(r0, i_xyz_t);
            }
            if ((std::abs(r_t.x - edge) > FP_PRECISION) && u.x != 0) {
              this_d = (edge - r_t.x) / u.x;
              if (this_d < d) {
                if (u.x > 0) {
                  lattice_trans = {1, 0, 0};
                } else {
                  lattice_trans = {-1, 0, 0};
                }
                d = this_d;
              }
            }

        // Top and bottom sides
        if (n_axial_>1) {

          double z = r0.z;
          double z0;
          //double z0 {copysign(0.5 * discretez_(ijk0[2]), u.z)};
          if (u.z > 0) {
        	   z0 = heightz_(ijk0[2] + 1);
          }
          else{
        	   z0 = heightz_(ijk0[2]);}

          if ((std::abs(z - z0) > FP_PRECISION) && u.z != 0) {
            this_d = (z0 - z) / u.z;
            if (this_d < d) {
              d = this_d;
              if (u.z > 0) {
                lattice_trans = {0, 0, 1};
              } else {
                lattice_trans = {0, 0, -1};
              }
              d = this_d;
            }
          }
        }


    double distance = d;
    if (are_valid_indices(ijk0)){
    	bins.push_back(get_bin_from_indices(ijk0));
    	lengths.push_back(distance / total_distance);
    }

    // Translate the starting coordintes by the distance to the oncoming mesh
    // surface.
    bool now_in_hex {in_hex(r0)};

    r0 += distance * u;

    ijk0[0] = ijk0[0] + lattice_trans[0];

    ijk0[1] = ijk0[1] + lattice_trans[1];

    ijk0[2] = ijk0[2] + lattice_trans[2];

    if ((now_in_hex) && !(in_hex(r0))){
    	break;
    }
  }
}
 void HexMesh::surface_bins_crossed(const Particle* p, std::vector<int>& bins) const
 {

	 constexpr int MAX_SEARCH_ITER = 100;


	   // ========================================================================
	   // Determine if the track intersects the tally mesh.

	   // Copy the starting and ending coordinates of the particle.  Offset these
	   // just a bit for the purposes of determining if there was an intersection
	   // in case the mesh surfaces coincide with lattice/geometric surfaces which
	   // might produce finite-precision errors.
	   Position last_r {p->r_last_};
	   Position r {p->r()};
	   Direction u {p->u()};

	   Position r0 = last_r + TINY_BIT*u;
	   Position r1 = r - TINY_BIT*u;

	   // Determine indices for starting and ending location.
	   int n = n_dimension_;
	   int ijk0[n], ijk1[n];
	   bool start_in_mesh;
	   bool finallycross;
	   bool end_in_mesh;
	   int i_bin;
	   crossing(r0,r1,&ijk0[0],&ijk1[0],finallycross,start_in_mesh,end_in_mesh);

      // std :: cout << "START POINT IS : " << ijk0[0] << " " << ijk0[1] << " "<< ijk0[2] << " " << std ::endl;
      // std :: cout << "FINISH POINT IS : " << ijk1[0] << " " << ijk1[1] << " "<< ijk1[2] << " " << std ::endl;
	   // Check if the track intersects any part of the mesh.
	   if (!start_in_mesh && !end_in_mesh) {
	     if (!finallycross) return;
	   }

	   // ========================================================================
	   // Figure out which mesh cell to tally.

	   // Copy the un-modified coordinates the particle direction.
	   r0 = last_r;
	   r1 = r;
	   // Compute the length of the entire track.
	   double total_distance = (r1 - r0).norm();

	   // The particle does not start in the mesh.  Note that we nudged the
	      // start and end coordinates by a TINY_BIT each so we will have
	      // difficulty resolving tracks that are less than 2*TINY_BIT in length.
	      // If the track is that short, it is also insignificant so we can
	      // safely ignore it in the tallies.
	    if (total_distance < 2*TINY_BIT) return;
	   //
	   // Compute the direction on the hexagonal basis.
	    double beta_dir = u.x*0.5  + u.y * std::sqrt(3.0) / 2.0;
	    double gamma_dir = u.x*0.5  - u.y * std::sqrt(3.0) / 2.0;

	     // Note that hexagonal lattice distance calculations are performed
	     // using the particle's coordinates relative to the neighbor lattice
	     // cells, not relative to the particle's current cell.  This is done
	     // because there is significant disagreement between neighboring cells
	     // on where the lattice boundary is due to finite precision issues.
	   while (true) {
	     // ========================================================================
	     // Compute the length of the track segment in the each mesh cell and return
         double this_d;
	     if (std::equal(ijk0, ijk0+n, ijk1)) {
	       // The track ends in this cell.  Use the particle end location rather
	       // than the mesh surface.
	       break;
	     }

	     // The track exits this cell.  Determine the distance to the closest mesh
	     // surface.
	     // Upper-right and lower-left sides.
	     double d {INFTY};
	             std::array<int, 3> lattice_trans;
	             double edge = -copysign(0.5*width_[0], beta_dir);  // Oncoming edge
	             Position r_t;
	             if (beta_dir > 0) {
	               const std::array<int, 3> i_xyz_t {ijk0[0], ijk0[1]+1, ijk0[2]};
	               r_t = get_local_position(r0, i_xyz_t);
	             } else {
	               const std::array<int, 3> i_xyz_t {ijk0[0], ijk0[1]-1, ijk0[2]};
	               r_t = get_local_position(r0, i_xyz_t);
	             }
	             double beta = r_t.x / 2.0 + r_t.y * std::sqrt(3.0) / 2.0;
	         if ((std::abs(beta - edge) > FP_PRECISION) && beta_dir != 0) {
	           d = (edge - beta) / beta_dir;
	           if (beta_dir > 0) {
	             lattice_trans = {0, 1, 0};
	             i_bin = 2;
	           } else {
	             lattice_trans = {0, -1, 0};
	             i_bin = 4;
	           }
	         }

	         // Lower-right and upper-left sides.
	                edge = -copysign(0.5*width_[0], gamma_dir);
	                if (gamma_dir > 0) {
	                    const std::array<int, 3> i_xyz_t {ijk0[0]+1, ijk0[1]-1, ijk0[2]};
	                    r_t = get_local_position(r0, i_xyz_t);
	                  } else {
	                    const std::array<int, 3> i_xyz_t {ijk0[0]-1, ijk0[1]+1, ijk0[2]};
	                    r_t = get_local_position(r0, i_xyz_t);
	                  }
	                  double gamma = r_t.x / 2.0 - r_t.y * std::sqrt(3.0) / 2.0;
	         if ((std::abs(gamma - edge) > FP_PRECISION) && gamma_dir != 0) {
	           double this_d = (edge - gamma) / gamma_dir;
	           if (this_d < d) {
	             if (gamma_dir > 0) {
	               lattice_trans = {1, -1, 0};
	               i_bin = 6;
	             } else {
	               lattice_trans = {-1, 1, 0};
	               i_bin = 8;
	             }
	             d = this_d;
	           }
	         }

	         edge = -copysign(0.5*width_[0], u.x);
	                    if (u.x > 0) {
	                      const std::array<int, 3> i_xyz_t {ijk0[0]+1, ijk0[1], ijk0[2]};
	                      r_t = get_local_position(r0, i_xyz_t);
	                    } else {
	                      const std::array<int, 3> i_xyz_t {ijk0[0]-1, ijk0[1], ijk0[2]};
	                      r_t = get_local_position(r0, i_xyz_t);
	                    }
	                    if ((std::abs(r_t.x - edge) > FP_PRECISION) && u.x != 0) {
	                      this_d = (edge - r_t.x) / u.x;
	           if (this_d < d) {
	             if (u.x > 0) {
	               lattice_trans = {1, 0, 0};
	               i_bin = 10;
	             } else {
	               lattice_trans = {-1, 0, 0};
	               i_bin = 12;
	             }
	             d = this_d;
	           }
	         }

	         // Top and bottom sides
	                    if (n_axial_>1) {

	                    	 double z = r0.z;
	                    	 double z0;
	                    	 //double z0 {copysign(0.5 * discretez_(ijk0[2]), u.z)};
	                    	 if (u.z > 0) {
	                    	       z0 = heightz_(ijk0[2] + 1);
	                    	  }
	                    	  else{
	                    	       z0 = heightz_(ijk0[2]);}

	                      if ((std::abs(z - z0) > FP_PRECISION) && u.z != 0) {
	                        this_d = (z0 - z) / u.z;
	                        if (this_d < d) {
	                          d = this_d;
	                          if (u.z > 0) {
	                            lattice_trans = {0, 0, 1};
	                            i_bin = 14;
	                          } else {
	                            lattice_trans = {0, 0, -1};
	                            i_bin = 16;
	                          }
	                          d = this_d;
	                        }
	                      }
	                    }


	     double distance = d;
	     if (are_valid_indices(ijk0)){
	    	 int lpv {get_bin_from_indices(ijk0)};
	    	 //lpv = shape_[0];

	    	 bins.push_back(4*4*get_bin_from_indices(ijk0) + i_bin - 1);

	     }
	     // Translate the starting coordintes by the distance to the oncoming mesh
	     // surface.
	     bool now_in_hex {in_hex(r0)};

	     r0 += distance * u;

	     ijk0[0] = ijk0[0] + lattice_trans[0];

	     ijk0[1] = ijk0[1] + lattice_trans[1];

	     ijk0[2] = ijk0[2] + lattice_trans[2];

	     if (are_valid_indices(ijk0)){

	    	 bins.push_back(4*4*get_bin_from_indices(ijk0) + i_bin - 2);


	    	     }

	     if ((now_in_hex) && !(in_hex(r0))){
	     	break;
	     }

	   }
 }

  //==============================================================================
  Position
  HexMesh::get_local_position(Position r, const std::array<int, 3> i_xyz)
  const
  {

   // DR for OX implementation

  	  // x_l = x_g - (center + pitch_x*index_a + pitch_y*sin(30)*index_y)
  	  	r.x -= (center_.x + (i_xyz[0] - n_rings_ + 1) * width_[0]
  	  	            + (i_xyz[1] - n_rings_ + 1) * width_[0] / 2.0);
  	  // x_l = x_g - (center + pitch_y*cos(30)*index_y)
  	    r.y -= center_.y + std::sqrt(3.0)/2.0 * (i_xyz[1] - n_rings_ + 1) * width_[0];

  	  //  r.z -= center_.z + (i_xyz[2] + 0.5)* width_[2];
  	 // r.z -= center_.z + heightz_(i_xyz[2]); !!!!!!!!!
  	  r.z -=  heightz_(i_xyz[2]);

  return r;
  }

//==============================================================================

  bool
  HexMesh::are_valid_indices(int *i_xyz) const
  {
    return ((i_xyz[0] >= 0) && (i_xyz[1] >= 0) && (i_xyz[2] >= 0)
            && (i_xyz[0] < 2*n_rings_-1) && (i_xyz[1] < 2*n_rings_-1)
            && (i_xyz[0] + i_xyz[1] > n_rings_-2)
            && (i_xyz[0] + i_xyz[1] < 3*n_rings_-2)
            && (i_xyz[2] < n_axial_));
  }

  //==============================================================================
  bool
  HexMesh::layinline(Position bpoint,Position epoint, Position inpoint) const
  {
	double dx {std::abs(epoint.x-bpoint.x)};
	double k;
	if (dx > TINY_BIT) {
		k = (epoint.y-bpoint.y)/(epoint.x-bpoint.x);
	}
	else
	{
		k = 0;
	}
	if (k >= 0) {
		if ((inpoint.x >= bpoint.x) && (epoint.x >= inpoint.x  )) {
			if ((inpoint.y >= bpoint.y) && (epoint.y >= inpoint.y  )) {
			    return true;
			}
		}
		if ((inpoint.x <= bpoint.x) && (epoint.x <= inpoint.x  )) {
					if ((inpoint.y <= bpoint.y) && (epoint.y <= inpoint.y  )) {
					    return true;
					}
				}
	}

	if (k <= 0) {
			if ((inpoint.x >= bpoint.x) && (epoint.x >= inpoint.x  )) {
				if ((inpoint.y <= bpoint.y) && (epoint.y <= inpoint.y  )) {
				    return true;
				}
			}
			if ((inpoint.x <= bpoint.x) && (epoint.x <= inpoint.x  )) {
						if ((inpoint.y >= bpoint.y) && (epoint.y >= inpoint.y  )) {
						    return true;
						}
					}
		}
   return false;
  }

  //==============================================================================
    bool
	HexMesh::iscross(xt::xarray<Position> &firstline,xt::xarray<Position> &secondline) const
    {
    	double a1,b1,c1,a2,b2,c2,det,dx,dy,xi,yi,zi;
        Position ipoint;
    	if (firstline(0).x != firstline(1).x){
    		a1 = -(firstline(1).y - firstline(0).y)/(firstline(1).x - firstline(0).x);
    		b1 = 1;
    		c1 = (firstline(1).x*firstline(0).y - firstline(1).y*firstline(0).x)/(firstline(1).x - firstline(0).x);
    	}
    	else
    	{
    		a1 = 1;
    		b1 = 1;
    		c1 = firstline(0).x;
    	}
    	if (secondline(0).x != secondline(1).x){
    	    		a2 = -(secondline(1).y - secondline(0).y)/(secondline(1).x - secondline(0).x);
    	    		b2 = 1;
    	    		c2 = (secondline(1).x*secondline(0).y - secondline(1).y*secondline(0).x)/(secondline(1).x - secondline(0).x);
    	    	}
    	else
    	{
    	    		a2 = 1;
    	    		b2 = 1;
    	    		c2 = secondline(0).x;
    	}
    	det = a1*b2-a2*b1;
    	if (std::abs(det) < TINY_BIT){
    	   return false;
    	}
    	else {
    		dx = (c1*b2 - c2*b1);
    	    dy = (a1*c2 - a2*c1);
    		xi = dx/det;
    		yi = dy/det;
    		ipoint.x = xi;
    		ipoint.y = yi;
    		if (n_axial_ > 1){

    			if (secondline(0).x != secondline(1).x){

    				ipoint.z = secondline(0).z + (xi - secondline(0).x)/(secondline(1).x - secondline(0).x)*(secondline(1).z - secondline(0).z);
    				if ((ipoint.z > heightz_(n_axial_)) || (ipoint.z < heightz_(0))){
    					return false;
    				}
    			}
    			else if (secondline(0).y != secondline(1).y){
    				ipoint.z = secondline(0).z + (yi - secondline(0).y)/(secondline(1).y - secondline(0).y)*(secondline(1).z - secondline(0).z);
    				if ((ipoint.z > heightz_(n_axial_)) || (ipoint.z < heightz_(0))){
    				    					return false;
    			}
    			}
    			else
    			{return false;}

    		}
	        ipoint.z = 0.0;

    	}

       if (layinline(firstline(0),firstline(1),ipoint)){
    	   if (layinline(secondline(0),secondline(1),ipoint)){
    		   return true;
    	   }
       }
       return false;
    }
    bool
	HexMesh::iscross_z(xt::xarray<Position> &line) const {
    	if (((line(0).z < heightz_(0)) && ((line(1).z > heightz_(0)))) || ((line(1).z < heightz_(0)) && ((line(0).z > heightz_(0))))){
    		return true;
    	}
    	if (((line(0).z > heightz_(n_axial_)) && ((line(1).z < heightz_(n_axial_)))) || ((line(1).z > heightz_(n_axial_)) && ((line(0).z < heightz_(n_axial_))))){
    	    return true;
    	}
    	return false;

    }
    //==============================================================================
    bool
	HexMesh::in_outer_hex(int number,int *ixyz) const
    {
    	switch(number){
    	case(0):
    			if ((ixyz[0]+ixyz[1]==n_rings_ - 2) && (ixyz[1] <= n_rings_-1) && (ixyz[1] >=0)) {return true;}
    	case(1):
				if ((ixyz[0]==2*n_rings_-1) && (ixyz[1] <= n_rings_-1) && (ixyz[1] >=0)) {return true;}
    	case(2):
    			if ((ixyz[1]==-1) && (ixyz[0] < 2*n_rings_-1) && (ixyz[0] >=n_rings_- 2)) {return true;}
    	case(3):
    			if ((ixyz[0]+ixyz[1]==3*n_rings_ + 2) && (ixyz[1] <= n_rings_-1) && (ixyz[1] >=0)) {return true;}
    	case(4):
    			if ((ixyz[0]==-1) && (ixyz[1] <= n_rings_-2) && (ixyz[1] >= 2*n_rings_- 1)) {return true;}
    	case(5):
    	    	if ((ixyz[1]==2*n_rings_-1) && (ixyz[0] <= n_rings_-1) && (ixyz[0] >=0)) {return true;}



    	}
    }
    //==============================================================================
    bool
	HexMesh::in_hex(Position P) const
    {
    	double u {P.x*sqrt(3.0) + P.y};
    	double w {-P.x*sqrt(3.0) + P.y};
    	double y {P.y};
    	double z {P.z};
    	if (n_axial_ > 1){

    		return ((in_hexagon(P)) && (z > heightz_(0))&& (z < heightz_(n_axial_)));

    	}
    	else
    	{
    		return in_hexagon(P);
    	}

    }

//==============================================================================
    bool
	   HexMesh::in_hexagon(Position P) const{

    	double u {P.x*sqrt(3.0) + P.y};
    	double w {-P.x*sqrt(3.0) + P.y};
    	double y {P.y};

    	return ((-_hpp < u) && (u < _hpp) && (-_hpp < w) && (w < _hpp) && (-_hpp/2 < y) && (y < _hpp/2));
    }


//==============================================================================
  int HexMesh::get_bin(Position r) const
  {


    // Determine indices
    int ijk[n_dimension_];
    bool in_mesh;
    get_indices(r, ijk, &in_mesh);
    if (!in_mesh) return -1;

    // Convert indices to bin
    return get_bin_from_indices(ijk);
  }

  int HexMesh::get_bin_from_indices(const int* ijk) const
  {
    switch (n_dimension_) {
      case 2:
        return  (2*n_rings_-1) * ijk[1] + ijk[0];
      case 3:
        return (ijk[2])*(2*n_rings_-1)*(2*n_rings_-1) + (2*n_rings_-1) * ijk[1] + ijk[0];
      default:
        throw std::runtime_error{"Invalid number of mesh dimensions"};
    }
  }

  void HexMesh::get_indices(Position r, int* ijk, bool* in_mesh) const
  {
    // Find particle in mesh
    *in_mesh = true;

     float dbg1;
     float dbg2;
     // Offset the xyz by the lattice center.
     Position r_o {r.x - center_.x, r.y - center_.y, r.z};
     //r_o.z -= center_.z;!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ATTENTION

     // Index the z direction.
     std::array<int, 3> out;
     //
     if (n_axial_ < 2){
    	 ijk[2] = 0;
     }

     double alpha = r_o.y - r_o.x * std::sqrt(3.0);
     dbg1=-alpha / (std::sqrt(3.0) * width_[0]);
     dbg2=r_o.y / 0.5*std::sqrt(3.0) * width_[0];
     ijk[0] = std::floor(-alpha / (std::sqrt(3.0) * width_[0]));
     ijk[1] = std::floor(r_o.y / (0.5*std::sqrt(3.0) * width_[0]));


     // Add offset to indices (the center cell is (i_x, i_alpha) = (0, 0) but
     // the array is offset so that the indices never go below 0).
     ijk[0] += n_rings_-1;
     ijk[1] += n_rings_-1;

     // Calculate the (squared) distance between the particle and the centers of
     // the four possible cells.  Regular hexagonal tiles form a Voronoi
     // tessellation so the xyz should be in the hexagonal cell that it is closest
     // to the center of.  This method is used over a method that uses the
     // remainders of the floor divisions above because it provides better finite
     // precision performance.  Squared distances are used becasue they are more
     // computationally efficient than normal distances.
     int k {1};
     int k_min {1};
     double d_min {INFTY};
     for (int i = 0; i < 2; i++) {
       for (int j = 0; j < 2; j++) {
         const std::array<int, 3> i_xyz {ijk[0] + j, ijk[1] + i, 0};
         Position r_t = get_local_position(r, i_xyz);
         double d = r_t.x*r_t.x + r_t.y*r_t.y;
         if (d < d_min) {
           d_min = d;
           k_min = k;
         }
         k++;
       }
     }

     // Select the minimum squared distance which corresponds to the cell the
     // coordinates are in.
     if (k_min == 2) {
       ++ijk[0];
     } else if (k_min == 3) {
       ++ijk[1];
     } else if (k_min == 4) {
       ++ijk[0];
       ++ijk[1];
     }

     if (n_axial_ > 1){

     if (r_o.z <= heightz_(0) ) {
    	 ijk[2] = -1;
     }
     else if (r_o.z >= heightz_(n_axial_) ) {
    	 ijk[2] = n_axial_;
     }
     else{
    	 for (int it = 0; it!=n_axial_;it++){
    		 if ((r_o.z > heightz_(it)) && (r_o.z < heightz_(it+1))) {
    			 ijk[2] = it;
    			 break;
    		 }
    	 }
     }

     }

     if (!are_valid_indices(ijk)) {*in_mesh = false;}
  }

  void HexMesh::get_indices_from_bin(int bin, int* ijk) const
  {
      int nx {2*n_rings_ - 1};
	  int ny {2*n_rings_ - 1};
	  int iz {bin / (nx * ny)};
	  int iy {(bin - nx * ny * iz) / nx};
	  int ix {bin - nx * ny * iz - nx * iy};
	  ijk[0] = ix;
	  ijk[1] = iy;
	  ijk[2] = iz;
  }

  bool HexMesh::intersects(Position r0, Position r1) const
  {
    switch(n_dimension_) {
      case 1:
        return true;//intersects_1d(r0, r1);
      case 2:
        return true;//intersects_2d(r0, r1);
      case 3:
        return true;//intersects_3d(r0, r1);
      default:
        throw std::runtime_error{"Invalid number of mesh dimensions."};
    }
  }
//
void HexMesh::crossing(Position r0,Position r1,int* posA,int* posB,bool& finallycross,bool& startinmesh, bool& endinmesh) const
  {
	  finallycross=false;
	  get_indices(r0 , posA, &startinmesh);
	  get_indices(r1 , posB, &endinmesh);
	  xt::xarray<Position> crossline({2});
	  crossline(0) = r0;
	  crossline(1) = r1;
	  for (int i = 0; i < 6; i++) {
		  xt::xarray<Position> line6 {xt::view(lines,xt::all(),i)};
		  if (iscross(line6, crossline)){

			      finallycross = true;

			  if ((!startinmesh) && (!endinmesh) && (in_outer_hex(i,posA)) && (in_outer_hex(i,posB))) {

				  finallycross = false;
			  }

		  }
	  }
	  if (!finallycross){

		  if ((!startinmesh) && (!endinmesh)){

			  if ((in_hex(r0)) && (in_hex(r1))){

				  if (posA != posB){

					  finallycross = true;
				  }
			  }
			  else if (n_axial_>1){
				  if ((iscross_z(crossline)) && (in_hexagon(crossline(0))) && (in_hexagon(crossline(1)))){

					  finallycross = true;

				  }
			  }
		  }

	  }

  }
void HexMesh::to_hdf5(hid_t group) const
{
  hid_t mesh_group = create_group(group, "mesh " + std::to_string(id_));
  std :: cout << "\n STATEPONT shape size  " << shape_.size() << " shape DIM1 "<<shape_(1)<< " shape DIM2 "<<shape_(2)<< " shape DIM3 "<<shape_(3)<< "\n";
  write_dataset(mesh_group, "type", "regular");
  write_dataset(mesh_group, "dimension", shape_);
  write_dataset(mesh_group, "lower_left", lower_left_);
  write_dataset(mesh_group, "upper_right", upper_right_);
  write_dataset(mesh_group, "width", width_);

  close_group(mesh_group);
}

xt::xarray<double>
HexMesh::count_sites(const std::vector<Particle::Bank>& bank,
  bool* outside) const
{
  // Determine shape of array for counts
  std::size_t m = xt::prod(shape_)();
  std::vector<std::size_t> shape = {m};

  // Create array of zeros
  xt::xarray<double> cnt {shape, 0.0};
  bool outside_ = false;

  for (const auto& site : bank) {
    // determine scoring bin for entropy mesh
    int mesh_bin = get_bin(site.r);

    // if outside mesh, skip particle
    if (mesh_bin < 0) {
      outside_ = true;
      continue;
    }

    // Add to appropriate bin
    cnt(mesh_bin) += site.wgt;
  }

  // Create copy of count data
  int total = cnt.size();
  double* cnt_reduced = new double[total];

#ifdef OPENMC_MPI
  // collect values from all processors
  MPI_Reduce(cnt.data(), cnt_reduced, total, MPI_DOUBLE, MPI_SUM, 0,
    mpi::intracomm);

  // Check if there were sites outside the mesh for any processor
  if (outside) {
    MPI_Reduce(&outside_, outside, 1, MPI_C_BOOL, MPI_LOR, 0, mpi::intracomm);
  }
#else
  std::copy(cnt.data(), cnt.data() + total, cnt_reduced);
  if (outside) *outside = outside_;
#endif

  // Adapt reduced values in array back into an xarray
  auto arr = xt::adapt(cnt_reduced, total, xt::acquire_ownership(), shape);
  xt::xarray<double> counts = arr;

  return counts;
}
//==============================================================================
// C API functions
//==============================================================================

//! Extend the meshes array by n elements
extern "C" int
openmc_extend_meshes(int32_t n, int32_t* index_start, int32_t* index_end)
{
  if (index_start) *index_start = model::meshes.size();
  for (int i = 0; i < n; ++i) {
    model::meshes.push_back(std::make_unique<RectMesh>());
  }
  if (index_end) *index_end = model::meshes.size() - 1;

  return 0;
}

//! Return the index in the meshes array of a mesh with a given ID
extern "C" int
openmc_get_mesh_index(int32_t id, int32_t* index)
{
  auto pair = model::mesh_map.find(id);
  if (pair == model::mesh_map.end()) {
    set_errmsg("No mesh exists with ID=" + std::to_string(id) + ".");
    return OPENMC_E_INVALID_ID;
  }
  *index = pair->second;
  return 0;
}

// Return the ID of a mesh
extern "C" int
openmc_mesh_get_id(int32_t index, int32_t* id)
{
  if (index < 0 || index >= model::meshes.size()) {
    set_errmsg("Index in meshes array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *id = model::meshes[index]->id_;
  return 0;
}

//! Set the ID of a mesh
extern "C" int
openmc_mesh_set_id(int32_t index, int32_t id)
{
  if (index < 0 || index >= model::meshes.size()) {
    set_errmsg("Index in meshes array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  model::meshes[index]->id_ = id;
  model::mesh_map[id] = index;
  return 0;
}

//! Get the dimension of a mesh
extern "C" int
openmc_mesh_get_dimension(int32_t index, int** dims, int* n)
{
  if (index < 0 || index >= model::meshes.size()) {
    set_errmsg("Index in meshes array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *dims = model::meshes[index]->shape_.data();
  *n = model::meshes[index]->n_dimension_;
  return 0;
}

//! Set the dimension of a mesh
extern "C" int
openmc_mesh_set_dimension(int32_t index, int n, const int* dims)
{
  if (index < 0 || index >= model::meshes.size()) {
    set_errmsg("Index in meshes array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  // Copy dimension
  std::vector<std::size_t> shape = {static_cast<std::size_t>(n)};
  auto& m = model::meshes[index];
  m->shape_ = xt::adapt(dims, n, xt::no_ownership(), shape);
  m->n_dimension_ = m->shape_.size();

  return 0;
}

//! Get the mesh parameters
extern "C" int
openmc_mesh_get_params(int32_t index, double** ll, double** ur, double** width, int* n)
{
  if (index < 0 || index >= model::meshes.size()) {
    set_errmsg("Index in meshes array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  auto& m = model::meshes[index];
  if (m->lower_left_.dimension() == 0) {
    set_errmsg("Mesh parameters have not been set.");
    return OPENMC_E_ALLOCATE;
  }

  *ll = m->lower_left_.data();
  *ur = m->upper_right_.data();
  *width = m->width_.data();
  *n = m->n_dimension_;
  return 0;
}

//! Set the mesh parameters
extern "C" int
openmc_mesh_set_params(int32_t index, int n, const double* ll, const double* ur,
                       const double* width)
{
  if (index < 0 || index >= model::meshes.size()) {
    set_errmsg("Index in meshes array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  auto& m = model::meshes[index];
  std::vector<std::size_t> shape = {static_cast<std::size_t>(n)};
  if (ll && ur) {
    m->lower_left_ = xt::adapt(ll, n, xt::no_ownership(), shape);
    m->upper_right_ = xt::adapt(ur, n, xt::no_ownership(), shape);
    m->width_ = (m->upper_right_ - m->lower_left_) / m->shape_;
  } else if (ll && width) {
    m->lower_left_ = xt::adapt(ll, n, xt::no_ownership(), shape);
    m->width_ = xt::adapt(width, n, xt::no_ownership(), shape);
    m->upper_right_ = m->lower_left_ + m->shape_ * m->width_;
  } else if (ur && width) {
    m->upper_right_ = xt::adapt(ur, n, xt::no_ownership(), shape);
    m->width_ = xt::adapt(width, n, xt::no_ownership(), shape);
    m->lower_left_ = m->upper_right_ - m->shape_ * m->width_;
  } else {
    set_errmsg("At least two parameters must be specified.");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  return 0;
}

//==============================================================================
// Non-member functions
//==============================================================================

void read_meshes(pugi::xml_node root)
{
	std :: cout << "Reading meshes";
	write_message("Reading meshes", 5);
	for (auto node : root.children("mesh")) {
	    // Read mesh and add to vector
		std :: cout << "Rect meshes\n";
		write_message("Rect meshes", 5);
		  model::meshes.push_back(std::make_unique<RectMesh>(node));
		  model::mesh_map[model::meshes.back()->id_] = model::meshes.size() - 1;
	  }
	  //
	  for (auto node : root.children("hexmesh")) {
		  std :: cout << "Hex meshes\n";
	      // Read mesh and add to vector
		  write_message("Hex meshes", 5);
	  	  model::meshes.push_back(std::make_unique<HexMesh>(node));
	  	  model::mesh_map[model::meshes.back()->id_] = model::meshes.size() - 1;
	    }
}

void meshes_to_hdf5(hid_t group)
{
  // Write number of meshes
  hid_t meshes_group = create_group(group, "meshes");
  int32_t n_meshes = model::meshes.size();
  write_attribute(meshes_group, "n_meshes", n_meshes);

  if (n_meshes > 0) {
    // Write IDs of meshes
    std::vector<int> ids;
    for (const auto& m : model::meshes) {
      m->to_hdf5(meshes_group);
      ids.push_back(m->id_);
    }
    write_attribute(meshes_group, "ids", ids);
  }

  close_group(meshes_group);
}

void free_memory_mesh()
{
  model::meshes.clear();
  model::mesh_map.clear();
}

extern "C" int n_meshes() { return model::meshes.size(); }

} // namespace openmc
