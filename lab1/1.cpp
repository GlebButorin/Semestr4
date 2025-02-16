#include <set>
#include <gmsh.h>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    gmsh::initialize();
    double lc = 0.1;

    double R = 0.1; // Major radius
    double r[2] = {0.02, 0.04}; // Minor radiuses of thorus-hole and thorus
    int circles = 20; // Number of points around the major circle
    int points = 20; // Number of points around the minor circle

    std::vector<int> SurfaceLoops;
    for (unsigned k = 0; k < 2; ++k) {
    std::vector<int> Points;
    for (int i = 0; i < circles; ++i) {
        double majorAngle = 2 * M_PI * i / circles;
        for (int j = 0; j < points; ++j) {
            double angle = 2 * M_PI * j / points;
            double x = (R + r[k] * cos(angle)) * cos(majorAngle);
            double y = (R + r[k] * cos(angle)) * sin(majorAngle);
            double z = r[k] * sin(angle);
            Points.push_back(gmsh::model::geo::addPoint(x, y, z, lc));
        }
    }

    std::vector<int> lines;
    for (int i = 0; i < circles; ++i) {
        for (int j = 0; j < points; ++j) {
            int p1 = Points[i * points + j];
            int p2 = Points[i * points + (j + 1) % points];
            int p3 = Points[((i + 1) % circles) * points + j];

            // line "up"
            lines.push_back(gmsh::model::geo::addLine(p1, p2));
            // line "right"
            lines.push_back(gmsh::model::geo::addLine(p1, p3));
        }
    }

    //std::cout << "done lines" << std::endl;

    std::vector<int> PlaneSurfaces;
    for (int i = 0; i < circles; ++i) {
        for (int j = 0; j < points; ++j) {
            std::vector<int> loopLines;
            // Counter-clockwise order
            int sign = 1;
            if (k == 0) {
                sign = -1;
            }

            loopLines.push_back(-lines[(i * points + j) * 2 + 0] * sign); 
            loopLines.push_back(lines[(i * points + j) * 2 + 1] * sign);
            loopLines.push_back(lines[(((i + 1) % circles) * points + j) * 2 + 0] * sign);
            loopLines.push_back(-lines[(i * points + (j + 1) % points) * 2 + 1] * sign);
            int curveLoopID = gmsh::model::geo::addCurveLoop(loopLines);
            PlaneSurfaces.push_back(gmsh::model::geo::addPlaneSurface({curveLoopID}));
        }
    }

    //std::cout << "done surfaces" << std::endl;


    SurfaceLoops.push_back(gmsh::model::geo::addSurfaceLoop(PlaneSurfaces));
    std::cout << SurfaceLoops[k]<< " " << k << std::endl;}

    gmsh::model::geo::addVolume(SurfaceLoops);

    gmsh::model::geo::synchronize();

    gmsh::model::mesh::generate(3);

    gmsh::write("torus.msh");

    std::set<std::string> args(argv, argv + argc);
    if (!args.count("-nopopup")) gmsh::fltk::run();

    gmsh::finalize();

    return 0;
}