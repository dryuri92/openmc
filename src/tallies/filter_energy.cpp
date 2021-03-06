#include "openmc/tallies/filter_energy.h"

#include "openmc/capi.h"
#include "openmc/constants.h"  // For F90_NONE
#include "openmc/mgxs_interface.h"
#include "openmc/search.h"
#include "openmc/settings.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// EnergyFilter implementation
//==============================================================================

void
EnergyFilter::from_xml(pugi::xml_node node)
{
  bins_ = get_node_array<double>(node, "bins");
  n_bins_ = bins_.size() - 1;

  // In MG mode, check if the filter bins match the transport bins.
  // We can save tallying time if we know that the tally bins match the energy
  // group structure.  In that case, the matching bin index is simply the group
  // (after flipping for the different ordering of the library and tallying
  // systems).
  if (!settings::run_CE) {
    if (n_bins_ == data::num_energy_groups) {
      matches_transport_groups_ = true;
      for (auto i = 0; i < n_bins_ + 1; i++) {
        if (data::rev_energy_bins[i] != bins_[i]) {
          matches_transport_groups_ = false;
          break;
        }
      }
    }
  }
}

void
EnergyFilter::get_all_bins(const Particle* p, int estimator, FilterMatch& match)
const
{
  if (p->g_ != F90_NONE && matches_transport_groups_) {
    if (estimator == ESTIMATOR_TRACKLENGTH) {
      match.bins_.push_back(data::num_energy_groups - p->g_);
    } else {
      match.bins_.push_back(data::num_energy_groups - p->g_last_);
    }
    match.weights_.push_back(1.0);

  } else {
    // Get the pre-collision energy of the particle.
    auto E = p->E_last_;

    // Bin the energy.
    if (E >= bins_.front() && E <= bins_.back()) {
      auto bin = lower_bound_index(bins_.begin(), bins_.end(), E);
      match.bins_.push_back(bin);
      match.weights_.push_back(1.0);
    }
  }
}

void
EnergyFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", bins_);
}

std::string
EnergyFilter::text_label(int bin) const
{
  std::stringstream out;
  out << "Incoming Energy [" << bins_[bin] << ", " << bins_[bin+1] << ")";
  return out.str();
}

//==============================================================================
// EnergyoutFilter implementation
//==============================================================================

void
EnergyoutFilter::get_all_bins(const Particle* p, int estimator,
                              FilterMatch& match) const
{
  if (p->g_ != F90_NONE && matches_transport_groups_) {
    match.bins_.push_back(data::num_energy_groups - p->g_);
    match.weights_.push_back(1.0);

  } else {
    if (p->E_ >= bins_.front() && p->E_ <= bins_.back()) {
      auto bin = lower_bound_index(bins_.begin(), bins_.end(), p->E_);
      match.bins_.push_back(bin);
      match.weights_.push_back(1.0);
    }
  }
}

std::string
EnergyoutFilter::text_label(int bin) const
{
  std::stringstream out;
  out << "Outgoing Energy [" << bins_[bin] << ", " << bins_[bin+1] << ")";
  return out.str();
}

//==============================================================================
// C-API functions
//==============================================================================

extern"C" int
openmc_energy_filter_get_bins(int32_t index, double** energies, int32_t* n)
{
  // Make sure this is a valid index to an allocated filter.
  if (int err = verify_filter(index)) return err;

  // Get a pointer to the filter and downcast.
  const auto& filt_base = model::tally_filters[index].get();
  auto* filt = dynamic_cast<EnergyFilter*>(filt_base);

  // Check the filter type.
  if (!filt) {
    set_errmsg("Tried to get energy bins on a non-energy filter.");
    return OPENMC_E_INVALID_TYPE;
  }

  // Output the bins.
  *energies = filt->bins_.data();
  *n = filt->bins_.size();
  return 0;
}

extern "C" int
openmc_energy_filter_set_bins(int32_t index, int32_t n, const double* energies)
{
  // Make sure this is a valid index to an allocated filter.
  if (int err = verify_filter(index)) return err;

  // Get a pointer to the filter and downcast.
  const auto& filt_base = model::tally_filters[index].get();
  auto* filt = dynamic_cast<EnergyFilter*>(filt_base);

  // Check the filter type.
  if (!filt) {
    set_errmsg("Tried to set energy bins on a non-energy filter.");
    return OPENMC_E_INVALID_TYPE;
  }

  // Update the filter.
  filt->bins_.clear();
  filt->bins_.resize(n);
  for (int i = 0; i < n; i++) filt->bins_[i] = energies[i];
  filt->n_bins_ = n - 1;
  return 0;
}

}// namespace openmc
