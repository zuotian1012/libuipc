#include <app/app.h>
#include <uipc/uipc.h>
#include <uipc/constitution/neo_hookean_shell.h>
#include <uipc/common/timer.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace
{
int probe_int(const char* name, int fallback)
{
    if(const char* value = std::getenv(name))
        return std::atoi(value);
    return fallback;
}

double probe_double(const char* name, double fallback)
{
    if(const char* value = std::getenv(name))
        return std::atof(value);
    return fallback;
}
}  // namespace

TEST_CASE("94_contact_density_probe", "[fem][research]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    const int layers  = probe_int("UIPC_PROBE_LAYERS", 2);
    const int frames  = probe_int("UIPC_PROBE_FRAMES", 40);
    const int contact = probe_int("UIPC_PROBE_CONTACT", 1);
    const int timers  = probe_int("UIPC_PROBE_TIMERS", 0);
    const int grid    = probe_int("UIPC_PROBE_GRID", 8);
    const double initial_down_speed = probe_double("UIPC_PROBE_INITIAL_DOWN_SPEED", 0.0);
    REQUIRE(layers >= 1);
    REQUIRE(layers <= 8);
    REQUIRE(frames >= 1);
    REQUIRE(frames <= 200);
    REQUIRE((contact == 0 || contact == 1));
    REQUIRE((timers == 0 || timers == 1));
    REQUIRE(grid >= 4);
    REQUIRE(grid <= 32);
    REQUIRE(initial_down_speed >= 0.0);
    REQUIRE(initial_down_speed <= 20.0);

    if(timers == 1)
        Timer::enable_all();
    GlobalTimer global_timer{"Contact Density Probe"};
    global_timer.set_as_current();

    auto   output_path = AssetDir::output_path(UIPC_RELATIVE_SOURCE_FILE);
    Engine engine{"cuda", output_path};
    World  world{engine};

    auto config                               = test::Scene::default_config();
    config["gravity"]                         = Vector3{0, -9.8, 0};
    config["contact"]["enable"]               = contact == 1;
    config["contact"]["constitution"]         = "ipc";
    config["contact"]["friction"]["enable"]   = false;
    config["line_search"]["max_iter"]         = 8;
    config["linear_system"]["tol_rate"]       = 1e-3;
    config["extras"]["strict_mode"]["enable"] = true;

    Scene           scene{config};
    NeoHookeanShell shell;
    scene.contact_tabular().default_model(0.5, 1.0_GPa);
    auto default_contact = scene.contact_tabular().default_element();
    auto object          = scene.objects().create("cloth stack");

    const int        N       = grid;
    const Float      spacing = 0.5 / N;
    vector<Vector3i> faces;
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            int v00 = i * (N + 1) + j;
            int v10 = (i + 1) * (N + 1) + j;
            int v01 = i * (N + 1) + j + 1;
            int v11 = (i + 1) * (N + 1) + j + 1;
            faces.push_back(Vector3i{v00, v10, v11});
            faces.push_back(Vector3i{v00, v11, v01});
        }
    }

    vector<S<SimplicialComplexSlot>> slots;
    for(int layer = 0; layer < layers; ++layer)
    {
        vector<Vector3> vertices;
        for(int i = 0; i <= N; ++i)
            for(int j = 0; j <= N; ++j)
                vertices.push_back(Vector3{i * spacing, 0.08 + layer * 0.04, j * spacing});

        auto mesh = trimesh(vertices, faces);
        label_surface(mesh);
        shell.apply_to(mesh, ElasticModuli2D::youngs_poisson(1.0_MPa, 0.49));
        auto velocity = mesh.vertices().find<Vector3>(builtin::velocity);
        REQUIRE(velocity);
        for(auto& value : view(*velocity))
            value = Vector3{0, -initial_down_speed, 0};
        default_contact.apply_to(mesh);
        auto [slot, rest_slot] = object->geometries().create(mesh);
        slots.push_back(slot);
    }
    if(contact == 1)
        object->geometries().create(ground(0.0));

    world.init(scene);
    REQUIRE(world.is_valid());
    std::ofstream probe_rows;
    if(const char* path = std::getenv("UIPC_PROBE_CSV_PATH"))
    {
        probe_rows.open(path);
        REQUIRE(probe_rows.good());
    }
    for(int i = 0; i < frames; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        world.advance();
        REQUIRE(world.is_valid());
        world.retrieve();
        const auto elapsed = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
        const auto stats = engine.frame_stats();
        REQUIRE(stats.at("completed").get<bool>());
        REQUIRE(stats.at("converged").get<bool>());

        Float                 min_y        = std::numeric_limits<Float>::max();
        Float                 min_gap      = std::numeric_limits<Float>::max();
        Float                 min_slab_gap = std::numeric_limits<Float>::max();
        vector<vector<Float>> layer_heights;
        vector<Float>         layer_mins;
        vector<Float>         layer_maxs;
        for(auto& slot : slots)
        {
            auto* mesh = slot->geometry().as<SimplicialComplex>();
            REQUIRE(mesh);
            layer_heights.emplace_back();
            Float layer_min = std::numeric_limits<Float>::max();
            Float layer_max = std::numeric_limits<Float>::lowest();
            for(const auto& position : view(mesh->positions()))
            {
                min_y     = std::min(min_y, position.y());
                layer_min = std::min(layer_min, position.y());
                layer_max = std::max(layer_max, position.y());
                layer_heights.back().push_back(position.y());
            }
            layer_mins.push_back(layer_min);
            layer_maxs.push_back(layer_max);
        }
        for(SizeT layer = 1; layer < layer_heights.size(); ++layer)
        {
            min_slab_gap =
                std::min(min_slab_gap, layer_mins[layer] - layer_maxs[layer - 1]);
            for(SizeT vertex = 0; vertex < layer_heights[layer].size(); ++vertex)
                min_gap = std::min(min_gap,
                                   layer_heights[layer][vertex]
                                       - layer_heights[layer - 1][vertex]);
        }
        if(layers == 1)
        {
            min_gap      = 0.0;
            min_slab_gap = 0.0;
        }
        if(contact == 1)
        {
            REQUIRE(min_y >= -1e-5);
            REQUIRE(min_gap >= -1e-5);
            REQUIRE(min_slab_gap >= -1e-5);
        }

        auto frame_row =
            fmt::format("PROBE_FRAME,{},{},{},{},{:.6f},{},{},{},{},{:.9f},{:.9f},{:.9f}\n",
                        layers,
                        contact,
                        grid,
                        world.frame(),
                        elapsed,
                        stats.at("newton_iterations").get<int>(),
                        stats.at("linear_solver_iterations").get<int>(),
                        stats.at("line_search_trials").get<int>(),
                        stats.at("last_line_search_alpha").get<double>(),
                        stats.at("last_ccd_toi").get<double>(),
                        min_y,
                        min_gap);
        auto slab_row =
            fmt::format("PROBE_SLAB,frame={},slab_gap={:.9f}\n", world.frame(), min_slab_gap);
        if(probe_rows.is_open())
        {
            probe_rows << frame_row << slab_row;
            REQUIRE(probe_rows.good());
        }
        else
        {
            fmt::print("{}{}", frame_row, slab_row);
        }
    }
    if(timers == 1)
    {
        if(const char* path = std::getenv("UIPC_PROBE_TIMER_RAW_PATH"))
        {
            std::ofstream stream{path};
            REQUIRE(stream.good());
            stream << global_timer.report_as_json().dump(2);
        }
        if(const char* path = std::getenv("UIPC_PROBE_TIMER_PATH"))
        {
            std::ofstream stream{path};
            REQUIRE(stream.good());
            stream << global_timer.report_merged_as_json().dump(2);
        }
    }
}
