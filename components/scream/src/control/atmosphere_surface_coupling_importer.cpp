#include "atmosphere_surface_coupling_importer.hpp"

#include "ekat/ekat_assert.hpp"
#include "ekat/util/ekat_units.hpp"

#include <array>

namespace scream
{
// =========================================================================================
SurfaceCouplingImporter::SurfaceCouplingImporter (const ekat::Comm& comm, const ekat::ParameterList& params)
  : AtmosphereProcess(comm, params)
{

}
// =========================================================================================
void SurfaceCouplingImporter::set_grids(const std::shared_ptr<const GridsManager> grids_manager)
{
  using namespace ekat::units;

  const auto& grid_name = m_params.get<std::string>("Grid");
  auto grid = grids_manager->get_grid(grid_name);

  m_num_cols = grid->get_num_local_dofs();      // Number of columns on this rank
  m_num_levs = grid->get_num_vertical_levels(); // Number of levels per column

  // The units of mixing ratio Q are technically non-dimensional.
  // Nevertheless, for output reasons, we like to see 'kg/kg'.
  auto Qunit = kg/kg;
  Qunit.set_string("kg/kg");
  Units nondim(0,0,0,0,0,0,0);
  auto Wm2 = W / m / m;
  Wm2.set_string("W/m2)");
  const auto m2 = m*m;

  // Define the different field layouts that will be used for this process
  using namespace ShortFieldTagsNames;

  FieldLayout scalar2d_layout      { {COL     }, {m_num_cols   } };
  FieldLayout surf_mom_flux_layout { {COL, CMP}, {m_num_cols, 2} };

  add_field<Updated>("sfc_alb_dir_vis",  scalar2d_layout,      nondim, grid_name);
  add_field<Updated>("sfc_alb_dir_nir",  scalar2d_layout,      nondim, grid_name);
  add_field<Updated>("sfc_alb_dif_vis",  scalar2d_layout,      nondim, grid_name);
  add_field<Updated>("sfc_alb_dif_nir",  scalar2d_layout,      nondim, grid_name);
  add_field<Updated>("surf_sens_flux",   scalar2d_layout,      W/m2,   grid_name);
  add_field<Updated>("surf_latent_flux", scalar2d_layout,      W/m2,   grid_name);
  add_field<Updated>("surf_lw_flux_up",  scalar2d_layout,      W/m2,   grid_name);
  add_field<Updated>("surf_mom_flux",    surf_mom_flux_layout, N/m2,   grid_name);
}
// =========================================================================================
void SurfaceCouplingImporter::setup_surface_coupling_data(const SCDataManager &sc_data_manager)
{
  m_num_imports = sc_data_manager.get_num_fields();

  EKAT_ASSERT_MSG(m_num_cols == sc_data_manager.get_field_size(),
                  "Error! Surface Coupling imports need to have size ncols.");

  m_cpl_imports_view_h = decltype(m_cpl_imports_view_h) (sc_data_manager.get_field_data_ptr(),
                                                         m_num_cols, m_num_imports);
  m_cpl_imports_view_d = Kokkos::create_mirror_view_and_copy(DefaultDevice(),
                                                             m_cpl_imports_view_h);

  m_column_info = decltype(m_column_info) ("m_info", m_num_imports);

  m_import_field_names = new name_t[m_num_imports];
  std::memcpy(m_import_field_names, sc_data_manager.get_field_name_ptr(), m_num_imports*32*sizeof(char));

  m_vector_components_view =
      decltype(m_vector_components_view) (sc_data_manager.get_field_vector_components_ptr(),
                                          m_num_imports);
  m_constant_multiple_view =
      decltype(m_constant_multiple_view) (sc_data_manager.get_field_constant_multiple_ptr(),
                                          m_num_imports);
}
// =========================================================================================
void SurfaceCouplingImporter::do_import()
{
  using policy_type = KokkosTypes<DefaultDevice>::RangePolicy;

  // Local copies, to deal with CUDA's handling of *this.
  const auto col_info = m_column_info;
  const auto cpl_imports_view_d = m_cpl_imports_view_d;
  const int num_cols = m_num_cols;
  const int num_imports = m_num_imports;

  std::cout << cpl_imports_view_d.extent(0) << "    " << cpl_imports_view_d.extent(1) << std::endl;

  // Deep copy cpl host array to device
  Kokkos::deep_copy(m_cpl_imports_view_d,m_cpl_imports_view_h);

  // Unpack the fields
  auto unpack_policy = policy_type(0,num_imports*num_cols);
  Kokkos::parallel_for(unpack_policy, KOKKOS_LAMBDA(const int& i) {
    const int ifield = i / num_cols;
    const int icol   = i % num_cols;

    const auto& info = col_info(ifield);

    auto offset = icol*info.col_stride + info.col_offset;

                         std::cout << "     " <<  m_column_info(ifield).col_stride
                         << "  " << m_column_info(ifield).col_offset
                         << "  " << m_column_info(ifield).constant_multiple << std::endl;

    info.data[offset] = cpl_imports_view_d(icol,ifield)*info.constant_multiple;
  });
}
// =========================================================================================
void SurfaceCouplingImporter::initialize_impl (const RunType /* run_type */)
{
  for (int i=0; i<m_num_imports; ++i) {

    std::string fname = m_import_field_names[i];
    if (fname == "unused") {

      // Do nothing

    } else {

      // Get the field and check that is valid
      Field field = get_field_out(fname);
      EKAT_REQUIRE_MSG (field.is_allocated(), "Error! Import field view has not been allocated yet.\n");

      // Set view data ptr
      m_column_info(i).data = field.get_internal_view_data<Real>();

      // Get column info from field utility function
      SurfaceCouplingUtils::get_col_info_for_surface_values(field.get_header_ptr(),
                                                            m_vector_components_view(i),
                                                            m_column_info(i).col_offset, m_column_info(i).col_stride);

      // Set constant multiple
      m_column_info(i).constant_multiple = m_constant_multiple_view(i);
    }
  }

  do_import();
}
// =========================================================================================
void SurfaceCouplingImporter::run_impl (const int /* dt */)
{
  do_import();
}
// =========================================================================================
void SurfaceCouplingImporter::finalize_impl()
{

}
// =========================================================================================
} // namespace scream
