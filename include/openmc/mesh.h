//! \file mesh.h
//! \brief Mesh types used for tallies, Shannon entropy, CMFD, etc.

#ifndef OPENMC_MESH_H
#define OPENMC_MESH_H

#include <memory> // for unique_ptr
#include <vector>
#include <unordered_map>

#include "hdf5.h"
#include "pugixml.hpp"
#include "xtensor/xarray.hpp"

#include "openmc/particle.h"
#include "openmc/position.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================
class RegularMesh;

enum class MeshType {
  rect, hex
};

namespace model {

extern std::vector<std::unique_ptr<RegularMesh>> meshes;
extern std::unordered_map<int32_t, int32_t> mesh_map;

} // namespace model


class RegularMesh
{
public:

	int id_ {-1};  //!< User-specified ID
	MeshType type_; //!<Type of regular mesh
	int n_dimension_; //!< Number of dimensions
	double volume_frac_; //!< Volume fraction of each mesh element
	xt::xarray<int> shape_; //!< Number of mesh elements in each dimension
	xt::xarray<double> width_; //!< Width of each mesh element
	RegularMesh();
	RegularMesh(pugi::xml_node node);

	virtual ~RegularMesh() {}

	virtual void bins_crossed(const Particle* p, std::vector<int>& bins,
	                    std::vector<double>& lengths) const= 0;


	virtual void surface_bins_crossed(const Particle* p, std::vector<int>& bins) const= 0;


	virtual  int get_bin(Position r) const=0;


	virtual int get_bin_from_indices(const int* ijk) const=0;


	virtual void get_indices(Position r, int* ijk, bool* in_mesh) const= 0;


	virtual  void get_indices_from_bin(int bin, int* ijk) const= 0;

	virtual void to_hdf5(hid_t group) const = 0;
//	virtual bool intersects(Position r0, Position r1) const=0;


	virtual  xt::xarray<double>
    count_sites(const std::vector<Particle::Bank>& bank,
      bool* outside) const=0;

	 xt::xarray<double> lower_left_; //!< Lower-left coordinates of mesh
	 xt::xarray<double> upper_right_; //!< Upper-right coordinates of mesh

};
//==============================================================================
//! Tessellation of n-dimensional Euclidean space by congruent squares or cubes
//==============================================================================
//==============================================================================
class RectMesh : public RegularMesh
{
public:
  // Constructors
  RectMesh();
  RectMesh(pugi::xml_node node) ;

  // Methods

  //! Determine which bins were crossed by a particle
  //!
  //! \param[in] p Particle to check
  //! \param[out] bins Bins that were crossed
  //! \param[out] lengths Fraction of tracklength in each bin
  void bins_crossed(const Particle* p, std::vector<int>& bins,
                    std::vector<double>& lengths) const;

  //! Determine which surface bins were crossed by a particle
  //!
  //! \param[in] p Particle to check
  //! \param[out] bins Surface bins that were crossed
  void surface_bins_crossed(const Particle* p, std::vector<int>& bins) const;

  //! Get bin at a given position in space
  //!
  //! \param[in] r Position to get bin for
  //! \return Mesh bin
  int get_bin(Position r) const;

  //! Get bin given mesh indices
  //!
  //! \param[in] Array of mesh indices
  //! \return Mesh bin
  int get_bin_from_indices(const int* ijk) const;

  //! Get mesh indices given a position
  //!
  //! \param[in] r Position to get indices for
  //! \param[out] ijk Array of mesh indices
  //! \param[out] in_mesh Whether position is in mesh
  void get_indices(Position r, int* ijk, bool* in_mesh) const;

  //! Get mesh indices corresponding to a mesh bin
  //!
  //! \param[in] bin Mesh bin
  //! \param[out] ijk Mesh indices
  void get_indices_from_bin(int bin, int* ijk) const;

  //! Check if a line connected by two points intersects the mesh
  //!
  //! \param[in] r0 Starting position
  //! \param[in] r1 Ending position
  //! \return Whether line connecting r0 and r1 intersects mesh
  bool intersects(Position r0, Position r1) const;

  //! Write mesh data to an HDF5 group
  //!
  //! \param[in] group HDF5 group
    void to_hdf5(hid_t group) const;

  //! Count number of bank sites in each mesh bin / energy bin
  //!
  //! \param[in] n Number of bank sites
  //! \param[in] bank Array of bank sites
  //! \param[in] n_energy Number of energies
  //! \param[in] energies Array of energies
  //! \param[out] Whether any bank sites are outside the mesh
  //! \return Array indicating number of sites in each mesh/energy bin
    xt::xarray<double>
    count_sites(const std::vector<Particle::Bank>& bank,
      bool* outside) const;





private:
  bool intersects_1d(Position r0, Position r1) const;
  bool intersects_2d(Position r0, Position r1) const;
  bool intersects_3d(Position r0, Position r1) const;
};
//
//==============================================================================
//! Tessellation of n-dimensional Euclidean space by hexagonal prism
//==============================================================================
class HexMesh : public RegularMesh
{
public:
  // Constructors
  //RegularMesh() = default;
  explicit HexMesh(pugi::xml_node node);

  // Methods

  //! Determine which bins were crossed by a particle
  //!
  //! \param[in] p Particle to check
  //! \param[out] bins Bins that were crossed
  //! \param[out] lengths Fraction of tracklength in each bin
  void bins_crossed(const Particle* p, std::vector<int>& bins,
                    std::vector<double>& lengths) const;




  //! Determine which surface bins were crossed by a particle
  //!
  //! \param[in] p Particle to check
  //! \param[out] bins Surface bins that were crossed
   void surface_bins_crossed(const Particle* p, std::vector<int>& bins) const;

  //! Get bin at a given position in space
  //!
  //! \param[in] r Position to get bin for
  //! \return Mesh bin
  int get_bin(Position r) const;

  //! Get bin given mesh indices
  //!
  //! \param[in] Array of mesh indices
  //! \return Mesh bin
  int get_bin_from_indices(const int* ijk) const;

  //! Get mesh indices given a position
  //!
  //! \param[in] r Position to get indices for
  //! \param[out] ijk Array of mesh indices
  //! \param[out] in_mesh Whether position is in mesh
  void get_indices(Position r, int* ijk, bool* in_mesh) const;

  //! Get mesh indices corresponding to a mesh bin
  //!
  //! \param[in] bin Mesh bin
  //! \param[out] ijk Mesh indices
  void get_indices_from_bin(int bin, int* ijk) const;

  //! Check if a line connected by two points intersects the mesh
  //!
  //! \param[in] r0 Starting position
  //! \param[in] r1 Ending position
  //! \return Whether line connecting r0 and r1 intersects mesh
  bool intersects(Position r0, Position r1) const;

  //! Write mesh data to an HDF5 group
  //!
  //! \param[in] group HDF5 group
  void to_hdf5(hid_t group) const;

  //! Count number of bank sites in each mesh bin / energy bin
  //!
  //! \param[in] n Number of bank sites
  //! \param[in] bank Array of bank sites
  //! \param[in] n_energy Number of energies
  //! \param[in] energies Array of energies
  //! \param[out] Whether any bank sites are outside the mesh
  //! \return Array indicating number of sites in each mesh/energy bin
  xt::xarray<double>
      count_sites(const std::vector<Particle::Bank>& bank,
        bool* outside) const;


  Position center_;               //!< Global center of lattice
  int n_rings_;                   //!< Number of radial tile positions
  int n_axial_;                   //!< Number of axial tile positions
  xt::xarray<double> discretez_;  //!< Discretization in axial direction
  xt::xarray<double> heightz_;    //!< Sizes of every height layer


private:
  int _number_bin;
  float _hpp;
  std::vector<size_t> _lineshapes = {6, 2};
  xt::xarray<Position> lines;
  void crossing(Position r0,Position r1,int* posA,int* posB,bool& finallycross,bool& startinmesh, bool& endinmesh) const;

  Position
    get_local_position(Position r, const std::array<int, 3> i_xyz) const;

  bool
  are_valid_indices(int *i_xyz) const;
  bool
    layinline(Position bpoint,Position epoint, Position inpoint) const;
  bool
    iscross(xt::xarray<Position> &firstline,xt::xarray<Position> &secondline) const;
  bool
    iscross_z(xt::xarray<Position> &line) const;
  bool
    in_outer_hex(int number,int *ixyz) const;
  bool
      in_hex(Position P) const;
  bool
      in_hexagon(Position P) const;

};
//==============================================================================
// Non-member functions
//==============================================================================

//! Read meshes from either settings/tallies
//
//! \param[in] root XML node
void read_meshes(pugi::xml_node root);

//! Write mesh data to an HDF5 group
//
//! \param[in] group HDF5 group
void meshes_to_hdf5(hid_t group);

void free_memory_mesh();

} // namespace openmc

#endif // OPENMC_MESH_H
