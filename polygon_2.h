#ifndef dealii_cgal_polygon_2_h
#define dealii_cgal_polygon_2_h

#include <deal.II/base/config.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/cgal/point_conversion.h>
#  include <CGAL/Polygon_2.h>



DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  using K         = CGAL::Exact_predicates_exact_constructions_kernel;

  using CGALPoint2 = CGAL::Point_2<K>;
  using CGALPolygon = CGAL::Polygon_2<K>;

  void dealii_cell_to_cgal_polygon(
    const typename Triangulation<2, 2>::cell_iterator &cell,
    const Mapping<2, 2>                               &mapping,
    CGALPolygon                                       &polygon)
  {
    //Note that this is accounting for curved boundary
    // not sure if needed but fitted geometry might be curved!
    // if not needed mapping object is obsolete then use cell->vertex()
    const auto &vertices = mapping.get_vertices(cell);
    polygon.clear();
    if(cell->reference_cell() == ReferenceCells::Triangle)
    {
      polygon.push_back(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2,2>(vertices[0]));
      polygon.push_back(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2,2>(vertices[1]));
      polygon.push_back(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2,2>(vertices[2]));
    }
    else if(cell->reference_cell() == ReferenceCells::Quadrilateral)
    {
      polygon.push_back(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2,2>(vertices[0]));
      polygon.push_back(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2,2>(vertices[1]));
      polygon.push_back(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2,2>(vertices[3]));
      polygon.push_back(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2,2>(vertices[2]));
    }
    else
    {
      DEAL_II_ASSERT_UNREACHABLE();
    }
  }

  void dealii_tria_to_cgal_polygon(
    const Triangulation<2, 2>        &tria,
    CGALPolygon                      &fitted_2D_mesh)
  {
    //Note: mapping is not considered here!!
    std::map<unsigned int, unsigned int> face_vertex_indices;

    for(const auto &cell : tria.active_cell_iterators())
    {
      for(const unsigned int i : cell->face_indices())
      {
        const typename Triangulation<2, 2>::face_iterator &face =
            cell->face(i);
        if(face->at_boundary())
        {
          //Note: for triangles automatically correct ordering
          if((i == 0 || i == 3) && cell->n_vertices() == 4 ){
            face_vertex_indices[face->vertex_index(1)] = face->vertex_index(0);
          }else{
            face_vertex_indices[face->vertex_index(0)] = face->vertex_index(1);
          }
        }
      }
    }
      
    const auto &vertices = tria.get_vertices();
    fitted_2D_mesh.clear();

    unsigned int current_index = face_vertex_indices.begin()->first;
    
    for(size_t i = 0; i < face_vertex_indices.size(); i++)
    {
      fitted_2D_mesh.push_back(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2,2>(vertices[current_index]));
      current_index = face_vertex_indices[current_index];
    }
  }
}

DEAL_II_NAMESPACE_CLOSE

#else

// Make sure the scripts that create the C++20 module input files have
// something to latch on if the preprocessor #ifdef above would
// otherwise lead to an empty content of the file.
DEAL_II_NAMESPACE_OPEN
DEAL_II_NAMESPACE_CLOSE

#endif
#endif