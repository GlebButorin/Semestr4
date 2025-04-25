#include <gmsh.h>
#include <vector>
#include <cmath>

int main(int argc, char **argv) {

    // Initialize GMSH
    gmsh::initialize(argc, argv);
    gmsh::model::add("Mountain");

    // Parameters
    const double k = 1000; // scaling factor
    const double L = 20.0 * k;       // Domain length (x-direction)
    const double W = 10.0 * k;       // Domain width (y-direction)
    const double H = 5.0 * k;        // Domain height (z-direction)
    const double mountain_h = 3.5 * k; // Mountain height
    const double mountain_r = 3.0 * k; // Mountain base radius

    // Create entities
    int box = gmsh::model::occ::addBox(0, 0, 0, L, W, H);
    int mountain = gmsh::model::occ::addCone(L/2, W/2, 0, 0, 0, mountain_h, mountain_r, 0);
    
    // Perform boolean operation
    std::vector<std::pair<int, int>> object = {{3, box}};
    std::vector<std::pair<int, int>> tool = {{3, mountain}};
    std::vector<std::pair<int, int>> out;
    std::vector<std::vector<std::pair<int, int>>> out_map = {out};
    gmsh::model::occ::cut(object, tool, out, out_map);
    gmsh::model::occ::synchronize();

    // Get all surfaces
    std::vector<std::pair<int, int>> surfaces;
    gmsh::model::getEntities(surfaces, 2);

    //get volumes
    std::vector<std::pair<int, int>> volumes;
    gmsh::model::getEntities(volumes, 3);

    // Classify boundaries
    std::vector<int> inlet, outlet, mountain_surf, ground, top, sides;
    for(auto &surf : surfaces) {
        double xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(surf.first, surf.second, xmin, ymin, zmin, xmax, ymax, zmax);
        
        if(std::abs(xmin) < 1e-6 && std::abs(xmax) < 1e-6) { // x ≈ 0
            inlet.push_back(surf.second);
        }
        else if(std::abs(xmin-L) < 1e-6 && std::abs(xmax-L) < 1e-6) { // x ≈ L
            outlet.push_back(surf.second);
        }
        else if(std::abs(zmin) < 1e-6 && std::abs(zmax) < 1e-6) { // z ≈ 0
            ground.push_back(surf.second);
        }
        else if(std::abs(zmin-H) < 1e-6 && std::abs(zmax-H) < 1e-6) { // z ≈ H
            top.push_back(surf.second);
        }
        else if((std::abs(ymin) < 1e-6 && std::abs(ymax) < 1e-6) or (std::abs(ymin - W) < 1e-6 && std::abs(ymax - W) < 1e-6)) { // sides (y=0 or y=W)
            sides.push_back(surf.second);
        }
        else { // mountain surface
            mountain_surf.push_back(surf.second);
        }
    }

    // Assign physical labels
    gmsh::model::addPhysicalGroup(2, inlet, 1, "inlet");
    gmsh::model::addPhysicalGroup(2, outlet, 2, "outlet");
    gmsh::model::addPhysicalGroup(2, mountain_surf, 3, "mountain");
    gmsh::model::addPhysicalGroup(2, ground, 4, "ground");
    gmsh::model::addPhysicalGroup(2, top, 5, "top");
    gmsh::model::addPhysicalGroup(2, sides, 6, "sides");
    gmsh::model::addPhysicalGroup(3, {volumes[0].second}, 7, "domain");

    // Set mesh size
    gmsh::option::setNumber("Mesh.MeshSizeMin", 0.45 * k);
    gmsh::option::setNumber("Mesh.MeshSizeMax", 0.9 * k);

    // Generate 3D mesh
    gmsh::model::mesh::generate(3);

    // Save mesh
    gmsh::write("Mountain.msh");

    // Launch GUI to preview
    gmsh::fltk::run();

    // Finalize
    gmsh::finalize();
    return 0;
}