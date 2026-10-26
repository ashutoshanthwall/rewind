// Python bindings for Rewind (steps 7.1 / 7.3): named indexes with time filtering.
// NumPy arrays go in by pointer (vectors are copied once into the engine's store); results come
// back as (ids int32, scores float32, report dict). Every call that touches an index releases
// the GIL; the index's own lock serialises calls to one index (collection.h).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "collection.h"
#include "storage.h"

namespace py = pybind11;
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

static py::tuple results(const CandidateArray& c, const SearchReport& r) {
    py::array_t<int> ids(c.size());
    py::array_t<float> scores(c.size());
    auto i = ids.mutable_unchecked<1>();
    auto s = scores.mutable_unchecked<1>();
    for (int j = 0; j < c.size(); j++) {
        i(j) = c[j].id;
        s(j) = c[j].score;
    }
    py::dict rep;
    rep["method"] = r.method;
    rep["in_range"] = r.inRange;
    rep["candidates"] = r.candidates;
    return py::make_tuple(ids, scores, rep);
}

static const float* queryPointer(const F32& q) {
    if (q.ndim() != 1) throw QueryError("query must be a 1-d float32 array");
    return q.data();
}

PYBIND11_MODULE(rewind_engine, m) {
    m.doc() = "Rewind C++ HNSW engine: named indexes, time-range filtering, save/load";

    // Engine errors become Python exceptions with the engine's message.
    static py::exception<Error> engineError(m, "EngineError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const Error& e) {
            py::set_error(engineError, e.what());
        }
    });

    m.def("float_count", &floatCount, "rows * dim as a checked 64-bit count");

    py::class_<NamedIndex>(m, "NamedIndex")
        .def(py::init<const std::string&, const std::string&>(), py::arg("name"), py::arg("model"))
        .def("load_vectors", &NamedIndex::loadVectors, py::arg("bin_path"), py::arg("manifest_path"),
             py::arg("dim"), py::arg("norm_tolerance"), py::call_guard<py::gil_scoped_release>())
        .def("set_vectors", [](NamedIndex& self, const F32& x, double tol) {
                 if (x.ndim() != 2) throw Error("vectors must be a 2-d float32 array");
                 long long rows = x.shape(0);
                 int dim = (int)x.shape(1);
                 const float* p = x.data();
                 py::gil_scoped_release release;
                 self.setVectors(dim, rows, p, tol);
             }, py::arg("vectors"), py::arg("norm_tolerance"))
        .def("set_times", [](NamedIndex& self, const F64& t) {
                 if (t.ndim() != 1) throw Error("times must be a 1-d float64 array");
                 self.setTimes(t.data(), t.shape(0));
             }, py::arg("times"))
        .def("build", [](NamedIndex& self, int M, int efConstruction, int efSearch, uint64_t seed) {
                 HnswParams p;
                 p.M = M;
                 p.efConstruction = efConstruction;
                 p.efSearch = efSearch;
                 p.seed = seed;
                 return self.build(p);
             }, py::arg("M"), py::arg("ef_construction"), py::arg("ef_search"), py::arg("seed"),
             py::call_guard<py::gil_scoped_release>())
        .def("save", &NamedIndex::save, py::arg("path"), py::call_guard<py::gil_scoped_release>())
        .def("load", &NamedIndex::load, py::arg("path"), py::call_guard<py::gil_scoped_release>())
        .def("search", [](NamedIndex& self, const F32& q, int k, int efSearch, std::optional<double> t0,
                          std::optional<double> t1, int oversample, double minFraction) {
                 if (t0.has_value() != t1.has_value()) throw QueryError("give both t0 and t1, or neither");
                 const float* p = queryPointer(q);
                 int dim = (int)q.shape(0);
                 FilterParams f;
                 f.oversample = oversample;
                 f.minFraction = minFraction;
                 CandidateArray out;
                 SearchReport rep;
                 {
                     py::gil_scoped_release release;
                     self.search(p, dim, k, efSearch, t0.has_value(), t0.value_or(0.0), t1.value_or(0.0), f,
                                 out, rep);
                 }
                 return results(out, rep);
             }, py::arg("query"), py::arg("k"), py::arg("ef_search"), py::arg("t0") = py::none(),
             py::arg("t1") = py::none(), py::arg("oversample"), py::arg("min_fraction"))
        .def("exact", [](NamedIndex& self, const F32& q, int k, std::optional<double> t0, std::optional<double> t1) {
                 if (t0.has_value() != t1.has_value()) throw QueryError("give both t0 and t1, or neither");
                 const float* p = queryPointer(q);
                 int dim = (int)q.shape(0);
                 CandidateArray out;
                 SearchReport rep;
                 {
                     py::gil_scoped_release release;
                     self.exact(p, dim, k, t0.has_value(), t0.value_or(0.0), t1.value_or(0.0), out, rep);
                 }
                 return results(out, rep);
             }, py::arg("query"), py::arg("k"), py::arg("t0") = py::none(), py::arg("t1") = py::none())
        .def_property_readonly("name", &NamedIndex::name)
        .def_property_readonly("model", &NamedIndex::model)
        .def_property_readonly("size", &NamedIndex::size)
        .def_property_readonly("dim", &NamedIndex::dim)
        .def_property_readonly("has_graph", &NamedIndex::hasGraph)
        .def_property_readonly("source_sha256", [](const NamedIndex& self) { return self.store().sourceSha256(); });

    py::class_<IndexCollection>(m, "IndexCollection")
        .def(py::init<>())
        .def("add", &IndexCollection::add, py::arg("name"), py::arg("model"), py::return_value_policy::reference_internal)
        .def("get", &IndexCollection::get, py::arg("name"), py::return_value_policy::reference_internal)
        .def("__contains__", &IndexCollection::has)
        .def("__len__", &IndexCollection::size)
        .def("names", [](const IndexCollection& self) {
            std::vector<std::string> out;
            for (int i = 0; i < self.size(); i++) out.push_back(self.nameAt(i));
            return out;
        });
}
