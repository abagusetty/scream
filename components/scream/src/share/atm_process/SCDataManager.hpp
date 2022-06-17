#ifndef SCREAM_SC_DATA_MANAGER_HPP
#define SCREAM_SC_DATA_MANAGER_HPP

#include "share/scream_types.hpp"
#include "ekat/ekat_assert.hpp"

namespace scream {

// This struct provides a intermediary between the AD and the SurfaceCouplingImport/Export AtmophereProcess
// classes, allowing SCREAM to access CPL data pointers and info about the fields.
struct SCDataManager {

  template<typename DevT, typename DataT>
  using view_1d = typename KokkosTypes<DevT>::template view_1d<DataT>;
  template<typename DevT, typename DataT>
  using view_2d = typename KokkosTypes<DevT>::template view_2d<DataT>;

  using name_t = char[32];

  SCDataManager() {}

  ~SCDataManager() = default;

  void setup_internals (const int field_size, const int num_fields, Real* field_data_ptr,
                        char* field_names, int* field_vector_components_ptr, Real* constant_multiple_ptr)
  {
    m_field_size = field_size;
    m_num_fields = num_fields;

    EKAT_ASSERT_MSG(field_data_ptr              !=nullptr, "Error! Ptr for field data is null.");
    EKAT_ASSERT_MSG(field_vector_components_ptr !=nullptr, "Error! Ptr for field vector components is null.");
    EKAT_ASSERT_MSG(field_names                 !=nullptr, "Error! Ptr for field names is null.");
    EKAT_ASSERT_MSG(constant_multiple_ptr       !=nullptr, "Error! Ptr for constant multiple is null.");
    m_field_data              = decltype(m_field_data)             (field_data_ptr,              m_field_size, m_num_fields);
    m_field_vector_components = decltype(m_field_vector_components)(field_vector_components_ptr, m_num_fields);
    m_field_constant_multiple = decltype(m_field_constant_multiple)(constant_multiple_ptr,       m_num_fields);

    // Fortran gives a 1d array of 32char strings. So let's memcpy the input char
    // strings into 2d char arrays. Each string is null-terminated (atm_mct_mod
    // makes sure of that).
    m_field_names = new name_t[num_fields];
    std::memcpy(m_field_names, field_names, num_fields*32*sizeof(char));
  }

  int get_field_size () const {
    return m_field_size;
  }

  int get_num_fields () const {
    return m_num_fields;
  }

  Real* get_field_data_ptr () const {
    return m_field_data.data();
  }

  char* get_field_name_ptr () const {
    return m_field_names[0];
  }

  int* get_field_vector_components_ptr () const {
    return m_field_vector_components.data();
  }

  Real* get_field_constant_multiple_ptr () const {
    return m_field_constant_multiple.data();
  }

protected:

  int m_field_size;
  int m_num_fields;
  int m_num_scream_fields;

  view_2d<HostDevice, Real> m_field_data;
  name_t*                   m_field_names;
  view_1d<HostDevice, int>  m_field_vector_components;
  view_1d<HostDevice, Real> m_field_constant_multiple;
};

} // scream

#endif // SCREAM_SC_DATA_MANAGER_HPP
