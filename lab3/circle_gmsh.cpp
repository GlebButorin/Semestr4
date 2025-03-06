#include <set>


#include <gmsh.h>

int main(int argc, char **argv)
{

  gmsh::initialize();

  gmsh::model::add("circle");

  double lc = 4e-2;
  unsigned N = 300;
  for (unsigned i =0; i < N; ++i) {
  gmsh::model::geo::addPoint(cos(2 * M_PI / N * i), sin(2 * M_PI / N * i), 0, lc, i + 1);}



  for (unsigned i = 0; i < N; ++i) {
    gmsh::model::geo::addLine(i + 1, (i + 1) % N + 1, i + 1);
  }

  std::vector<int> Lines;
  for (unsigned i = 0; i < N; ++i) {
    Lines.push_back(i + 1);
  }

  gmsh::model::geo::addCurveLoop(Lines, 1);


  gmsh::model::geo::addPlaneSurface({1}, 1);


  gmsh::model::geo::synchronize();

  gmsh::model::mesh::generate(2);

  gmsh::write("circle.msh");

  std::set<std::string> args(argv, argv + argc);
  if(!args.count("-nopopup")) gmsh::fltk::run();

  gmsh::finalize();

  return 0;
}