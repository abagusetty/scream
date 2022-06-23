#include "atmosphere_surface_coupling_importer.hpp"

#include "share/property_checks/field_within_interval_check.hpp"

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
  m_grid = grids_manager->get_grid(grid_name);

  m_num_cols = m_grid->get_num_local_dofs();      // Number of columns on this rank
  m_num_levs = m_grid->get_num_vertical_levels(); // Number of levels per column

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

  add_field<Computed>("sfc_alb_dir_vis",  scalar2d_layout,      nondim, grid_name);
  add_field<Computed>("sfc_alb_dir_nir",  scalar2d_layout,      nondim, grid_name);
  add_field<Computed>("sfc_alb_dif_vis",  scalar2d_layout,      nondim, grid_name);
  add_field<Computed>("sfc_alb_dif_nir",  scalar2d_layout,      nondim, grid_name);
  add_field<Computed>("surf_sens_flux",   scalar2d_layout,      W/m2,   grid_name);
  add_field<Computed>("surf_latent_flux", scalar2d_layout,      W/m2,   grid_name);
  add_field<Computed>("surf_lw_flux_up",  scalar2d_layout,      W/m2,   grid_name);
  add_field<Computed>("surf_mom_flux",    surf_mom_flux_layout, N/m2,   grid_name);

  // Number of fields SCREAM is expecting to import
  m_num_scream_imports = 9;
}
// =========================================================================================
  void SurfaceCouplingImporter::setup_surface_coupling_data(const SCDataManager &sc_data_manager)
{
  m_num_cpl_imports = sc_data_manager.get_num_fields();

  EKAT_ASSERT_MSG(m_num_scream_imports <= m_num_cpl_imports,
                  "Error! There are less imports in the SCDataManager (cpl imports) "
                  "than SCREAM imports.\n");
  EKAT_ASSERT_MSG(m_num_cols == sc_data_manager.get_field_size(),
                  "Error! Surface Coupling imports need to have size ncols.\n");

  m_cpl_imports_view_h = decltype(m_cpl_imports_view_h) (sc_data_manager.get_field_data_ptr(),
                                                         m_num_cols, m_num_cpl_imports);
  m_cpl_imports_view_d = Kokkos::create_mirror_view_and_copy(DefaultDevice(),
                                                             m_cpl_imports_view_h);
  m_import_field_names = new name_t[m_num_cpl_imports];
  std::memcpy(m_import_field_names, sc_data_manager.get_field_name_ptr(), m_num_cpl_imports*32*sizeof(char));

  m_vector_components_view =
      decltype(m_vector_components_view) (sc_data_manager.get_field_vector_components_ptr(),
                                          m_num_cpl_imports);
  m_constant_multiple_view =
      decltype(m_constant_multiple_view) (sc_data_manager.get_field_constant_multiple_ptr(),
                                          m_num_cpl_imports);

  // We only want to store column data for the fields SCREAM actually imports,
  // therefore we set the size to m_num_scream_imports.
  m_column_info = decltype(m_column_info) ("m_info", m_num_scream_imports);
}
// =========================================================================================
void SurfaceCouplingImporter::initialize_impl (const RunType /* run_type */)
{
  // Since we only store column data for SCREAM imports, there are two different indices,
  // one for SCREAM and one for CPL. In CPL data, the names of the unused fields should 
  // be marked "unused". At the end of the while loop we can check that the data coming 
  // in from CPL matches what SCREAM is expecting to import.
  Int i_cpl = 0;
  Int i_scream = 0; 
  while (i_cpl<m_num_cpl_imports) {

    std::string fname = m_import_field_names[i_cpl];
  
    if (fname == "unused") {
      
      // This field unused by SCREAM, only increment i_cpl
      ++i_cpl;
    
    } else {

      // This field is used by SCREAM. Note that m_column_info is indexed by i_scream,
      // data coming from the SCDataManager is indexed in i_cpl.  

      // Get the field and check that is valid
      Field field = get_field_out(fname);
      EKAT_REQUIRE_MSG (field.is_allocated(), "Error! Import field view has not been allocated yet.\n");

      // Set view data ptr
      m_column_info(i_scream).data = field.get_internal_view_data<Real>();

      // Get column info from field utility function
      SurfaceCouplingUtils::get_col_info_for_surface_values(field.get_header_ptr(),
                                                            m_vector_components_view(i_cpl),
                                                            m_column_info(i_scream).col_offset, m_column_info(i_scream).col_stride);
  
      // Set constant multiple
      m_column_info(i_scream).constant_multiple = m_constant_multiple_view(i_cpl);

      // Set index for referencing cpl data.
      m_column_info(i_scream).cpl_indx = i_cpl;

      ++i_scream;
      ++i_cpl;
    }
  }

  // Sanity check
  EKAT_REQUIRE_MSG(m_num_scream_imports == i_scream, "Error!: More or less scream imports than was expected.\n");
  EKAT_REQUIRE_MSG(m_num_cpl_imports == i_cpl, "Error!: More or less cpl imports than was expected.\n");

  // Set property checks for fields in this process                                                                                        
  add_precondition_check<FieldWithinIntervalCheck>(get_field_out("sfc_alb_dir_vis"),m_grid,0.0,1.0,true);
  add_precondition_check<FieldWithinIntervalCheck>(get_field_out("sfc_alb_dir_nir"),m_grid,0.0,1.0,true);
  add_precondition_check<FieldWithinIntervalCheck>(get_field_out("sfc_alb_dif_vis"),m_grid,0.0,1.0,true);
  add_precondition_check<FieldWithinIntervalCheck>(get_field_out("sfc_alb_dif_nir"),m_grid,0.0,1.0,true);
}
// =========================================================================================                                          
void SurfaceCouplingImporter::do_import()
{
  using policy_type = KokkosTypes<DefaultDevice>::RangePolicy;

  // Local copies, to deal with CUDA's handling of *this.                                                                             
  const auto col_info = m_column_info;
  const auto cpl_imports_view_d = m_cpl_imports_view_d;
  const int num_cols = m_num_cols;
  const int num_imports = m_num_scream_imports;

  // Deep copy cpl host array to device                                                                                               
  Kokkos::deep_copy(m_cpl_imports_view_d,m_cpl_imports_view_h);

  // Unpack the fields                                                                                                                
  auto unpack_policy = policy_type(0,num_imports*num_cols);
  Kokkos::parallel_for(unpack_policy, KOKKOS_LAMBDA(const int& i) {
    const int ifield = i / num_cols;
    const int icol   = i % num_cols;

    const auto& info = col_info(ifield);

    auto offset = icol*info.col_stride + info.col_offset;
    info.data[offset] = cpl_imports_view_d(icol,info.cpl_indx)*info.constant_multiple;
  });
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
