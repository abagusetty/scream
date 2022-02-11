#include <catch2/catch.hpp>

#include "control/atmosphere_driver.hpp"

#include "control/atmosphere_surface_coupling_importer.hpp"
#include "control/atmosphere_surface_coupling_exporter.hpp"
#include "share/grid/mesh_free_grids_manager.hpp"
#include "share/atm_process/atmosphere_process.hpp"
#include "share/scream_types.hpp"

#include "ekat/ekat_parse_yaml_file.hpp"

#include <iomanip>

namespace scream {

TEST_CASE("surface-coupling", "") {
  using namespace scream;
  using namespace scream::control;

  // Create a comm
  ekat::Comm atm_comm (MPI_COMM_WORLD);

  // Load ad parameter list
  std::string fname = "input.yaml";
  ekat::ParameterList ad_params("Atmosphere Driver");
  REQUIRE_NOTHROW ( parse_yaml_file(fname,ad_params) );

  // Time stepping parameters
  auto& ts              = ad_params.sublist("Time Stepping");
  const auto dt         = ts.get<int>("Time Step");
  const auto start_date = ts.get<std::vector<int>>("Start Date");
  const auto start_time = ts.get<std::vector<int>>("Start Time");
  const auto nsteps     = ts.get<int>("Number of Steps");
  const int ncols       = ad_params.sublist("Grids Manager").get<int>("Number of Global Columns");

  EKAT_ASSERT_MSG (dt>0, "Error! Time step must be positive.\n");

  util::TimeStamp t0 (start_date, start_time);
  EKAT_ASSERT_MSG (t0.is_valid(), "Error! Invalid start date.\n");

  // Need to register products in the factory *before* we create any atm process or grids manager.
  auto& proc_factory = AtmosphereProcessFactory::instance();
  auto& gm_factory = GridsManagerFactory::instance();
  proc_factory.register_product("SurfaceCouplingImporter",&create_atmosphere_process<SurfaceCouplingImporter>);
  proc_factory.register_product("SurfaceCouplingExporter",&create_atmosphere_process<SurfaceCouplingExporter>);
  gm_factory.register_product("Mesh Free",&create_mesh_free_grids_manager);

  // Create the driver
  AtmosphereDriver ad;

  // Setup views to test import/export. For this test we consider a random number of non-imported/exported
  // cpl fields (in addition to the required scream imports/exports), then assign a random number (no repeats)
  // to each field and later verify that imports and exports were done properly.
  const int num_scream_imports = 9;
  const int num_scream_exports = 17;
  std::srand(/*time(NULL)*/5);
  const int num_cpl_imports = num_scream_imports + (std::rand()%10);
  const int num_cpl_exports = num_scream_exports + (std::rand()%10);

  KokkosTypes<HostDevice>::view_2d<Real> import_data_view("import_data", ncols, num_cpl_imports);
  KokkosTypes<HostDevice>::view_1d<int>  import_vec_comps_view("import_vec_comps", num_cpl_imports);
  KokkosTypes<HostDevice>::view_1d<Real>  import_constant_multiple_view("import_constant_multiple_view", num_cpl_imports);
  char import_names[num_cpl_imports][32];

  KokkosTypes<HostDevice>::view_2d<Real> export_data_view("export_data", ncols, num_cpl_exports);
  KokkosTypes<HostDevice>::view_1d<int>  export_vec_comps_view("export_vec_comps", num_cpl_exports);
  KokkosTypes<HostDevice>::view_1d<Real>  export_constant_multiple_view("export_constant_multiple_view", num_cpl_exports);
  char export_names[num_cpl_exports][32];

  {
    std::cout << "Number of cpl imports: " << num_cpl_imports << " number of scream imports: " << num_scream_imports << std::endl;
    std::cout << "Number of cpl exports: " << num_cpl_exports << " number of scream exports: " << num_scream_exports << std::endl;

    Kokkos::deep_copy(import_data_view, 5.0);

    KokkosTypes<HostDevice>::view_1d<int>  tmp_import_vec_comps_view("tmp_import_vec_comps", num_cpl_imports);
    Kokkos::deep_copy(tmp_import_vec_comps_view, -1);
    KokkosTypes<HostDevice>::view_1d<Real>  tmp_import_constant_multiple_view("tmp_import_constant_multiple_view", num_cpl_imports);
    Kokkos::deep_copy(tmp_import_constant_multiple_view, 1);

    KokkosTypes<HostDevice>::view_1d<int>  tmp_export_vec_comps_view("tmp_export_vec_comps", num_cpl_exports);
    Kokkos::deep_copy(tmp_export_vec_comps_view, -1);
    KokkosTypes<HostDevice>::view_1d<Real>  tmp_export_constant_multiple_view("tmp_export_constant_multiple_view", num_cpl_exports);
    Kokkos::deep_copy(tmp_export_constant_multiple_view, 1);


    char tmp_import_names[num_cpl_imports][32];
    char tmp_export_names[num_cpl_exports][32];

    std::strcpy(tmp_import_names[0], "sfc_alb_dir_vis");
    std::strcpy(tmp_import_names[1], "sfc_alb_dir_nir");
    std::strcpy(tmp_import_names[2], "sfc_alb_dif_vis");
    std::strcpy(tmp_import_names[3], "sfc_alb_dif_nir");
    std::strcpy(tmp_import_names[4], "surf_mom_flux"), tmp_import_vec_comps_view(4) = 0, tmp_import_constant_multiple_view(4) = -1;
    std::strcpy(tmp_import_names[5], "surf_mom_flux"), tmp_import_vec_comps_view(5) = 1, tmp_import_constant_multiple_view(5) = -1;
    std::strcpy(tmp_import_names[6], "surf_sens_flux"),                                  tmp_import_constant_multiple_view(6) = -1;
    std::strcpy(tmp_import_names[7], "surf_latent_flux"),                                tmp_import_constant_multiple_view(7) = -1;
    std::strcpy(tmp_import_names[8], "surf_lw_flux_up"),                                 tmp_import_constant_multiple_view(8) = -1;
    for (int i=num_scream_imports; i<num_cpl_imports; ++i) std::strcpy(tmp_import_names[i], "unused");

    std::vector<int> import_order(num_cpl_imports);
    for (int i=0; i<num_cpl_imports; ++i) { import_order[i] = i; }
    std::random_shuffle(import_order.begin(), import_order.end());
    for (int i=0; i<num_cpl_imports; ++i) {
      std::strcpy(import_names[i], tmp_import_names[import_order[i]]);
      import_vec_comps_view(i) = tmp_import_vec_comps_view(import_order[i]);
      import_constant_multiple_view(i) = tmp_import_constant_multiple_view(import_order[i]);
    }

    std::strcpy(tmp_export_names[0], "Sa_z");
    std::strcpy(tmp_export_names[1], "horiz_winds"), tmp_export_vec_comps_view(1) = 0;
    std::strcpy(tmp_export_names[2], "horiz_winds"), tmp_export_vec_comps_view(2) = 1;
    std::strcpy(tmp_export_names[3], "T_mid");
    std::strcpy(tmp_export_names[4], "Sa_ptem");
    std::strcpy(tmp_export_names[5], "p_mid");
    std::strcpy(tmp_export_names[6], "qv");
    std::strcpy(tmp_export_names[7], "Sa_dens");
    std::strcpy(tmp_export_names[8], "Sa_pslv");
    std::strcpy(tmp_export_names[9], "Faxa_rainl");
    std::strcpy(tmp_export_names[10], "Faxa_snowl");
    std::strcpy(tmp_export_names[11], "sfc_flux_dir_nir");
    std::strcpy(tmp_export_names[12], "sfc_flux_dir_vis");
    std::strcpy(tmp_export_names[13], "sfc_flux_dif_nir");
    std::strcpy(tmp_export_names[14], "sfc_flux_dif_vis");
    std::strcpy(tmp_export_names[15], "sfc_flux_sw_net");
    std::strcpy(tmp_export_names[16], "sfc_flux_lw_dn");
    for (int i=num_scream_exports; i<num_cpl_exports; ++i) std::strcpy(tmp_export_names[i], "set_zero");

    std::vector<int> export_order(num_cpl_exports);
    for (int i=0; i<num_cpl_exports; ++i) { export_order[i] = i; }
    std::random_shuffle(export_order.begin(), export_order.end());
    for (int i=0; i<num_cpl_exports; ++i) {
      std::strcpy(export_names[i], tmp_export_names[export_order[i]]);
      export_vec_comps_view(i) = tmp_export_vec_comps_view(export_order[i]);
      export_constant_multiple_view(i) = tmp_export_constant_multiple_view(export_order[i]);
    }
  }

  ad.setup_sc_import(ncols, num_cpl_imports, import_data_view.data(),
                     import_names[0], import_vec_comps_view.data(),
                     import_constant_multiple_view.data());
  ad.setup_sc_export(ncols, num_cpl_exports, export_data_view.data(),
                     export_names[0], export_vec_comps_view.data(),
                     export_constant_multiple_view.data());

  // Init and run
  ad.initialize(atm_comm,ad_params,t0);

  if (atm_comm.am_i_root()) {
    printf("Start time stepping loop...       [  0%%]\n");
  }
  for (int i=0; i<nsteps; ++i) {
    ad.run(dt);
    if (atm_comm.am_i_root()) {
      std::cout << "  - Iteration " << std::setfill(' ') << std::setw(3) << i+1 << " completed";
      std::cout << "       [" << std::setfill(' ') << std::setw(3) << 100*(i+1)/nsteps << "%]\n";
    }
  }

  // TODO: get the field repo from the driver, and go get (one of)
  //       the output(s) of SurfaceCoupling, to check its numerical value (if possible)

  // Finalize 
  ad.finalize();

  // If we got here, we were able to run surface_coupling
  REQUIRE(true);
}

} // empty namespace
