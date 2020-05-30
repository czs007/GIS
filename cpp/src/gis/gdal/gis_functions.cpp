/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gis/gdal/gis_functions.h"

#include <omp.h>
#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <chrono>

#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/version.h"
#include "gis/gdal/arctern_geos.h"
#include "gis/gdal/geometry_visitor.h"
#include "gis/parser.h"
#include "utils/check_status.h"

namespace arctern {
namespace gis {
namespace gdal {

static int omp_parallelism = 0;
static int num_of_procs = 0;

// inline void* Wrapper_OGR_G_Centroid(void* geo) {
//   void* centroid = new OGRPoint();
//   OGR_G_Centroid(geo, centroid);
//   return centroid;
// }

template <typename T>
struct ArrayValue {
  bool is_null = false;
  T item_value;
};

template <typename T>
struct ChunkArrayIdx {
  int chunk_idx = 0;
  int array_idx = 0;
  bool is_null = false;
  T item_value;
};

struct WkbItem {
  const void* data_ptr;
  int wkb_size;
  OGRGeometry* ToGeometry() {
    if (data_ptr == nullptr) return nullptr;
    if (wkb_size <= 0) return nullptr;
    OGRGeometry* geo = nullptr;
    auto err_code = OGRGeometryFactory::createFromWkb(data_ptr, nullptr, &geo, wkb_size);
    if (err_code != OGRERR_NONE) return nullptr;
    return geo;
  }
};

inline OGRGeometry* Wrapper_createFromWkt(
    const std::shared_ptr<arrow::StringArray>& array, int idx) {
  if (array->IsNull(idx)) return nullptr;
  auto wkb_str = array->GetString(idx);

  if (parser::IsValidWkt(wkb_str.c_str()) == false) return nullptr;
  OGRGeometry* geo = nullptr;
  auto err_code = OGRGeometryFactory::createFromWkt(wkb_str.c_str(), nullptr, &geo);
  if (err_code != OGRERR_NONE) return nullptr;
  return geo;
}

inline OGRGeometry* Wrapper_createFromWkb(
    const std::shared_ptr<arrow::BinaryArray>& array, int idx) {
  if (array->IsNull(idx)) return nullptr;
  arrow::BinaryArray::offset_type wkb_size;
  auto data_ptr = array->GetValue(idx, &wkb_size);
  if (wkb_size <= 0) return nullptr;

  OGRGeometry* geo = nullptr;
  auto err_code = OGRGeometryFactory::createFromWkb(data_ptr, nullptr, &geo, wkb_size);
  if (err_code != OGRERR_NONE) return nullptr;
  return geo;
}

inline OGRGeometry* Wrapper_CurveToLine(OGRGeometry* geo, HasCurveVisitor* has_curve) {
  if (geo != nullptr) {
    has_curve->reset();
    geo->accept(has_curve);
    if (has_curve->has_curve()) {
      auto linear = geo->getLinearGeometry();
      OGRGeometryFactory::destroyGeometry(geo);
      return linear;
    }
  }
  return geo;
}

// inline char* Wrapper_OGR_G_ExportToWkt(OGRGeometry* geo) {
//   char* str;
//   auto err_code = OGR_G_ExportToWkt(geo, &str);
//   if (err_code != OGRERR_NONE) {
//     std::string err_msg =
//         "failed to export to wkt, error code = " + std::to_string(err_code);
//     throw std::runtime_error(err_msg);
//   }
//   return str;
// }


inline void AppendWkbNDR(arrow::BinaryBuilder& builder, const OGRGeometry* geo) {
  if (geo == nullptr) {
    builder.AppendNull();
  } else if (geo->IsEmpty() && (geo->getGeometryType() == wkbPoint)) {
    builder.AppendNull();
  } else {
    auto wkb_size = geo->WkbSize();
    auto wkb = static_cast<unsigned char*>(CPLMalloc(wkb_size));
    auto err_code = geo->exportToWkb(OGRwkbByteOrder::wkbNDR, wkb);
    if (err_code != OGRERR_NONE) {
      builder.AppendNull();
      // std::string err_msg =
      //     "failed to export to wkb, error code = " + std::to_string(err_code);
      // throw std::runtime_error(err_msg);
    } else {
      CHECK_ARROW(builder.Append(wkb, wkb_size));
    }
    CPLFree(wkb);
  }
}

template <typename T, typename Enable = void>
struct ChunkArrayBuilder {
  static constexpr int64_t CAPACITY = _ARROW_ARRAY_SIZE;  // Debug: 16M, Release 1G
  // static constexpr int64_t CAPACITY = 1024 * 1024 * 1024;
};

template <typename T>
struct ChunkArrayBuilder<
    T, typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value>::type> {
  T array_builder;
  int64_t array_size = 0;
};

inline std::shared_ptr<arrow::Array> AppendBoolean(
    ChunkArrayBuilder<arrow::BooleanBuilder>& builder, bool val) {
  std::shared_ptr<arrow::Array> array_ptr = nullptr;
  if (builder.array_size / 8 >= ChunkArrayBuilder<void>::CAPACITY) {
    CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
    builder.array_size = 0;
  }
  builder.array_builder.Append(val);
  ++builder.array_size;
  return array_ptr;
}

inline std::shared_ptr<arrow::Array> AppendDouble(
    ChunkArrayBuilder<arrow::DoubleBuilder>& builder, double val) {
  std::shared_ptr<arrow::Array> array_ptr = nullptr;
  if (builder.array_size + sizeof(val) > ChunkArrayBuilder<void>::CAPACITY) {
    CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
    builder.array_size = 0;
  }
  builder.array_builder.Append(val);
  builder.array_size += sizeof(val);
  return array_ptr;
}

inline std::shared_ptr<arrow::Array> AppendString(
    ChunkArrayBuilder<arrow::StringBuilder>& builder, std::string&& str_val) {
  std::shared_ptr<arrow::Array> array_ptr = nullptr;
  if (builder.array_size + str_val.size() > ChunkArrayBuilder<void>::CAPACITY) {
    CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
    builder.array_size = 0;
  }
  builder.array_size += str_val.size();
  CHECK_ARROW(builder.array_builder.Append(std::move(str_val)));
  return array_ptr;
}

inline std::shared_ptr<arrow::Array> AppendString(
    ChunkArrayBuilder<arrow::StringBuilder>& builder, const char* val) {
  if (val == nullptr) {
    builder.array_builder.AppendNull();
    return nullptr;
  } else {
    auto str_val = std::string(val);
    return AppendString(builder, std::move(str_val));
  }
}

inline std::shared_ptr<arrow::Array> AppendWkb(
    ChunkArrayBuilder<arrow::BinaryBuilder>& builder, const OGRGeometry* geo) {
  std::shared_ptr<arrow::Array> array_ptr = nullptr;
  if (geo == nullptr) {
    builder.array_builder.AppendNull();
  } else if (geo->IsEmpty() && (geo->getGeometryType() == wkbPoint)) {
    builder.array_builder.AppendNull();
  } else {
    auto wkb_size = geo->WkbSize();
    auto wkb = static_cast<unsigned char*>(CPLMalloc(wkb_size));
    auto err_code = geo->exportToWkb(OGRwkbByteOrder::wkbNDR, wkb);
    if (err_code != OGRERR_NONE) {
      builder.array_builder.AppendNull();
    } else {
      if (builder.array_size + wkb_size > ChunkArrayBuilder<void>::CAPACITY) {
        CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
        builder.array_size = 0;
      }
      CHECK_ARROW(builder.array_builder.Append(wkb, wkb_size));
      builder.array_size += wkb_size;
    }
    CPLFree(wkb);
  }
  return array_ptr;
}


bool GetArrayValue(const std::vector<std::shared_ptr<arrow::Array>>& chunk_array,
		  int index,
                  ChunkArrayIdx<double>& idx) {

int i = -1;
int accum = 0;
std::shared_ptr<arrow::Array> ptr = nullptr;
for (const auto & _ptr: chunk_array){
++i;
int cur_len = _ptr->length();
if (accum + cur_len > index) {
	idx.array_idx = index - accum; 
	break;
}
accum += cur_len;
}


  if (ptr->IsNull(idx.array_idx)) {
    idx.is_null = true;
    return true;
  }

  auto double_array =
      std::static_pointer_cast<arrow::DoubleArray>(ptr);
  idx.item_value = double_array->Value(idx.array_idx);
  idx.is_null = false;
  return true;
}

bool GetArrayValue(const std::vector<std::shared_ptr<arrow::Array>>& chunk_array,
		  int index,
                  ChunkArrayIdx<WkbItem>& idx) {

int i = -1;
int accum = 0;
std::shared_ptr<arrow::Array> ptr = nullptr;
for (const auto & _ptr: chunk_array){
++i;
int cur_len = _ptr->length();
if (accum + cur_len > index) {
	idx.array_idx = index - accum; 
	ptr = _ptr;
	break;
}
accum += cur_len;
}

  if (ptr->IsNull(idx.array_idx)) {
    idx.is_null = true;
    idx.item_value.data_ptr = nullptr;
    idx.item_value.wkb_size = 0;
    return true;
  }
  auto binary_array =
      std::static_pointer_cast<arrow::BinaryArray>(ptr);
  arrow::BinaryArray::offset_type wkb_size;
  auto data_ptr = binary_array->GetValue(idx.array_idx, &wkb_size);
  idx.item_value.data_ptr = data_ptr;
  idx.item_value.wkb_size = wkb_size;
  idx.is_null = (idx.item_value.wkb_size > 0);
  return true;

}

bool GetNextValue(const std::vector<std::shared_ptr<arrow::Array>>& chunk_array,
                  ChunkArrayIdx<WkbItem>& idx) {
  if (idx.chunk_idx >= (int)chunk_array.size()) return false;
  int len = chunk_array[idx.chunk_idx]->length();
  if (idx.array_idx >= len) {
    idx.chunk_idx++;
    idx.array_idx = 0;
    return GetNextValue(chunk_array, idx);
  }
  if (chunk_array[idx.chunk_idx]->IsNull(idx.array_idx)) {
    idx.array_idx++;
    idx.is_null = true;
    idx.item_value.data_ptr = nullptr;
    idx.item_value.wkb_size = 0;
    return true;
  }
  auto binary_array =
      std::static_pointer_cast<arrow::BinaryArray>(chunk_array[idx.chunk_idx]);
  arrow::BinaryArray::offset_type wkb_size;
  auto data_ptr = binary_array->GetValue(idx.array_idx, &wkb_size);
  idx.item_value.data_ptr = data_ptr;
  idx.item_value.wkb_size = wkb_size;
  idx.array_idx++;
  idx.is_null = (idx.item_value.wkb_size > 0);
  return true;
}

bool GetNextValue(const std::vector<std::shared_ptr<arrow::Array>>& chunk_array,
                  ChunkArrayIdx<double>& idx) {
  if (idx.chunk_idx >= (int)chunk_array.size()) return false;
  int len = chunk_array[idx.chunk_idx]->length();
  if (idx.array_idx >= len) {
    idx.chunk_idx++;
    idx.array_idx = 0;
    return GetNextValue(chunk_array, idx);
  }
  if (chunk_array[idx.chunk_idx]->IsNull(idx.array_idx)) {
    idx.array_idx++;
    idx.is_null = true;
    return true;
  }
  auto double_array =
      std::static_pointer_cast<arrow::DoubleArray>(chunk_array[idx.chunk_idx]);
  idx.item_value = double_array->Value(idx.array_idx);
  idx.array_idx++;
  idx.is_null = false;
  return true;
}

template <typename T>
bool GetNextValue(std::vector<std::vector<std::shared_ptr<arrow::Array>>>& array_list,
                  std::vector<ChunkArrayIdx<T>>& idx_list, bool& is_null) {
  auto ret_val = GetNextValue(array_list[0], idx_list[0]);
  is_null = idx_list[0].is_null;

  for (int i = 1; i < array_list.size(); ++i) {
    auto cur_val = GetNextValue(array_list[i], idx_list[i]);
    if (cur_val != ret_val) {
      throw std::runtime_error("incorrect input data");
    }
    is_null |= idx_list[i].is_null;
  }
  return ret_val;
}

void split_indexs(int total, int part, std::vector<std::pair<int, int>> & indexs);

void split_indexs(int total, int part, std::vector<std::pair<int, int>> & indexs){
	int reminder = total % part;
	int quote = total / part;
	

	int part1 = reminder;
	int quote1 = quote + 1;
	int part2 = part - reminder;
	int quote2 = quote;

	for( int i = 0; i < part1 ; i++ ) { 
		int start = i * quote1; 
		int end = ( i + 1 ) * quote1 - 1; 
		indexs.push_back(std::make_pair(start, end));
	}
	int offset = part1 * quote1;
	for (int i = 0; i < part2; ++i){
		int start = i * quote2 + offset; 
		int end = ( i + 1 ) * quote2 - 1 + offset; 
		indexs.push_back(std::make_pair(start, end));
	}
}

template <typename T, typename A>
typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value,
                        std::shared_ptr<typename arrow::Array>>::type
UnaryOp(const std::shared_ptr<arrow::Array>& array,
        std::function<bool(ArrayValue<A>&, OGRGeometry*)> op,
        std::function<void(T&, ArrayValue<A>&)>
	     append_op = nullptr) {

  auto wkb = std::static_pointer_cast<arrow::BinaryArray>(array);
  auto len = array->length();
  T builder;

  std::vector<ArrayValue<A>> array_values(len);
  //omp_set_dynamic(0); 
  int parallelism = get_parallelism();
  omp_set_num_threads(parallelism);

//  std::vector<std::pair<int, int>> indexs; 
//  split_indexs(len, parallelism, indexs);

  /*
  #pragma omp parallel for num_threads(parallelism)
  for (int i = 0; i <parallelism; ++i){
	int start = indexs[i].first;
	int end = indexs[i].second;
	for (int j = start; j<= end; ++j){
	auto geo = Wrapper_createFromWkb(wkb, j);
    	auto & item = array_values[j];
    	bool keep_geo = false;
	    if (geo == nullptr) {
	      item.is_null = true;
	    } else {
	      keep_geo = op(item, geo);
	    }
	    if (! keep_geo){
	       OGRGeometryFactory::destroyGeometry(geo);
	    }

	}

  }
  */

  #pragma omp parallel for num_threads(parallelism)
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkb(wkb, i);
    auto & item = array_values[i];
    bool keep_geo = false;
    if (geo == nullptr) {
      item.is_null = true;
    } else {
      keep_geo = op(item, geo);
    }
    if (! keep_geo){
       OGRGeometryFactory::destroyGeometry(geo);
    }
  }

  if (append_op) {
      for (auto & item: array_values){
          if (item.is_null){
		  builder.AppendNull();
	  }else{
		  append_op(builder, item);
	  }
      }
  }

  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

template <typename T, typename A>
typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value,
                        std::vector<std::shared_ptr<typename arrow::Array>>>::type
UnaryOp(const std::shared_ptr<arrow::Array>& array,
        std::function<bool(ArrayValue<A>&, OGRGeometry*)> op,
        std::function<std::shared_ptr<arrow::Array>(ChunkArrayBuilder<T>&, ArrayValue<A>&)>
	     append_op = nullptr
	    ) {
  auto wkb = std::static_pointer_cast<arrow::BinaryArray>(array);
  auto len = array->length();
  ChunkArrayBuilder<T> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;
  std::vector<ArrayValue<A>> array_values(len);

  omp_set_dynamic(0); 
  int parallelism = get_parallelism();
  omp_set_num_threads(parallelism);

  #pragma omp parallel for num_threads(parallelism)
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkb(wkb, i);
    bool keep_geo = false;
    auto & item = array_values[i];
    if (geo == nullptr) {
        item.is_null = true;
    } else {
        op(item, geo);
    }
    
    if (! keep_geo){
        OGRGeometryFactory::destroyGeometry(geo);
    }
    
  }

  if (append_op) {
      for (auto & item: array_values){
          if (item.is_null){
	      	builder.array_builder.AppendNull();
          }else{
		auto array_ptr = append_op(builder, item);
		if (array_ptr != nullptr) result_array.push_back(array_ptr);
	 }
      }
  }

  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

template <typename T, typename A>
typename std::enable_if<std::is_base_of<arrow::ArrayBuilder, T>::value,
                        std::vector<std::shared_ptr<typename arrow::Array>>>::type
BinaryOp(const std::vector<std::shared_ptr<typename arrow::Array>>& geo1,
         const std::vector<std::shared_ptr<typename arrow::Array>>& geo2,
         std::function<void(ArrayValue<A> &, OGRGeometry*, OGRGeometry*)>
             op,
         std::function<void(ArrayValue<A> &, OGRGeometry*, OGRGeometry*)>
             null_op = nullptr,
        std::function<std::shared_ptr<arrow::Array>(ChunkArrayBuilder<T>&, ArrayValue<A>&)>
	     append_op = nullptr
	     ) {

  std::vector<std::vector<std::shared_ptr<arrow::Array>>> array_list{geo1, geo2};
  ChunkArrayBuilder<T> builder;
  bool is_null;

  int total_geo = 0;
  std::vector<int> geo1_len_accum;
  for (const auto & item: geo1){
	total_geo += item->length(); 
	geo1_len_accum.push_back(total_geo);
  }

  std::vector<ChunkArrayIdx<WkbItem>> idx_list(2);

  std::vector<ArrayValue<A>> array_values(total_geo);

  omp_set_dynamic(0); 
  int parallelism = get_parallelism();
  omp_set_num_threads(parallelism);

  #pragma omp parallel for num_threads(parallelism)
  for (int i = 0; i < total_geo; ++i){
	ChunkArrayIdx<WkbItem> idx0;
	ChunkArrayIdx<WkbItem> idx1;
	GetArrayValue(array_list[0], i, idx0);
	GetArrayValue(array_list[1], i, idx1);

    auto ogr1 = idx0.item_value.ToGeometry();
    auto ogr2 = idx1.item_value.ToGeometry();
    if ((ogr1 == nullptr) && (ogr2 == nullptr)) {
	array_values[i].is_null=true;
    } else if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      if (null_op == nullptr) {
	array_values[i].is_null = true;
      } else {
	null_op(array_values[i], ogr1, ogr2);
      }
    } else {
	op(array_values[i], ogr1, ogr2);
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }

  std::vector<std::shared_ptr<arrow::Array>> result_array;
  if (append_op) {
      for (auto & item: array_values){
	if(item.is_null){
	   builder.array_builder.AppendNull();
	}else{
	   auto array_ptr = append_op(builder, item);
	   if (array_ptr != nullptr) result_array.push_back(array_ptr);
	}
      }
  }

  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

/************************ GEOMETRY CONSTRUCTOR ************************/

std::vector<std::shared_ptr<arrow::Array>> ST_Point(
    const std::vector<std::shared_ptr<arrow::Array>>& x_values_raw,
    const std::vector<std::shared_ptr<arrow::Array>>& y_values_raw) {
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> array_list{x_values_raw,
                                                                     y_values_raw};
  std::vector<ChunkArrayIdx<double>> idx_list(2);

  OGRPoint point;
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;
  bool is_null;

  while (GetNextValue(array_list, idx_list, is_null)) {
    if (is_null) {
      builder.array_builder.AppendNull();
    } else {
      point.setX(idx_list[0].item_value);
      point.setY(idx_list[1].item_value);
      auto array_ptr = AppendWkb(builder, &point);
      if (array_ptr != nullptr) result_array.push_back(array_ptr);
    }
  }

  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::vector<std::shared_ptr<arrow::Array>> ST_PolygonFromEnvelope(
    const std::vector<std::shared_ptr<arrow::Array>>& min_x_values,
    const std::vector<std::shared_ptr<arrow::Array>>& min_y_values,
    const std::vector<std::shared_ptr<arrow::Array>>& max_x_values,
    const std::vector<std::shared_ptr<arrow::Array>>& max_y_values) {
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> array_list{
      min_x_values, min_y_values, max_x_values, max_y_values};
  std::vector<ChunkArrayIdx<double>> idx_list(4);
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;
  bool is_null;
  OGRPolygon empty;

  while (GetNextValue(array_list, idx_list, is_null)) {
    if (is_null) {
      builder.array_builder.AppendNull();
    } else {
      if ((idx_list[0].item_value > idx_list[2].item_value) ||
          (idx_list[1].item_value > idx_list[3].item_value)) {
        auto array_ptr = AppendWkb(builder, &empty);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      } else {
        OGRLinearRing ring;
        ring.addPoint(idx_list[0].item_value, idx_list[1].item_value);
        ring.addPoint(idx_list[0].item_value, idx_list[3].item_value);
        ring.addPoint(idx_list[2].item_value, idx_list[3].item_value);
        ring.addPoint(idx_list[2].item_value, idx_list[1].item_value);
        ring.addPoint(idx_list[0].item_value, idx_list[1].item_value);
        ring.closeRings();
        OGRPolygon polygon;
        polygon.addRing(&ring);
        auto array_ptr = AppendWkb(builder, &polygon);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      }
    }
  }
  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::vector<std::shared_ptr<arrow::Array>> ST_GeomFromGeoJSON(
    const std::shared_ptr<arrow::Array>& json) {
  auto json_geo = std::static_pointer_cast<arrow::StringArray>(json);
  int len = json_geo->length();
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;

  for (int i = 0; i < len; ++i) {
    if (json_geo->IsNull(i)) {
      builder.array_builder.AppendNull();
    } else {
      auto str = json_geo->GetString(i);
      auto geo = (OGRGeometry*)OGR_G_CreateGeometryFromJson(str.c_str());
      if (geo != nullptr) {
        auto array_ptr = AppendWkb(builder, geo);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
        OGRGeometryFactory::destroyGeometry(geo);
      } else {
        builder.array_builder.AppendNull();
      }
    }
  }
  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::vector<std::shared_ptr<arrow::Array>> ST_GeomFromText(
    const std::shared_ptr<arrow::Array>& text) {
  auto geo = std::static_pointer_cast<arrow::StringArray>(text);
  auto len = geo->length();
  ChunkArrayBuilder<arrow::BinaryBuilder> builder;
  std::vector<std::shared_ptr<arrow::Array>> result_array;

  for (int i = 0; i < len; ++i) {
    auto ogr = Wrapper_createFromWkt(geo, i);
    if (ogr == nullptr) {
      builder.array_builder.AppendNull();
    } else {
      auto array_ptr = AppendWkb(builder, ogr);
      if (array_ptr != nullptr) result_array.push_back(array_ptr);
    }
    OGRGeometryFactory::destroyGeometry(ogr);
  }
  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::vector<std::shared_ptr<arrow::Array>> ST_AsText(
    const std::shared_ptr<arrow::Array>& wkb) {

  auto append_op = [](ChunkArrayBuilder<arrow::StringBuilder>& builder, ArrayValue<char*> & item) {
    auto result = AppendString(builder, item.item_value);
    CPLFree(item.item_value);
    return result;
  };

  auto op = [](ArrayValue<char*> & value, OGRGeometry* geo) {
    char* str;
    auto err_code = geo->exportToWkt(&str);
    value.is_null = err_code != OGRERR_NONE;
    value.item_value = str;
    return false;
  };

  return UnaryOp<arrow::StringBuilder, char*>(wkb, op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_AsGeoJSON(
    const std::shared_ptr<arrow::Array>& wkb) {

  auto append_op = [](ChunkArrayBuilder<arrow::StringBuilder>& builder, ArrayValue<char*>& item) {
    auto result = AppendString(builder, item.item_value);
    CPLFree(item.item_value);
    return result;
  };

  auto op = [](ArrayValue<char *> & value, OGRGeometry* geo) {
    char* str = geo->exportToJson();
    value.is_null = str == nullptr;
    value.item_value = str;
    return false;
   };

  return UnaryOp<arrow::StringBuilder,char *>(wkb, op, append_op);
}

/************************* GEOMETRY ACCESSOR **************************/
std::shared_ptr<arrow::Array> ST_IsValid(const std::shared_ptr<arrow::Array>& array) {

    auto start = std::chrono::high_resolution_clock::now();

  auto append_op = [](arrow::BooleanBuilder& builder, ArrayValue<bool>& item) {
	builder.Append(item.item_value);
  };

  auto op = [](ArrayValue<bool> & value, OGRGeometry* geo) {
    value.item_value = geo->IsValid() != 0;
    return false;
  };
  auto result = UnaryOp<arrow::BooleanBuilder, bool>(array, op, append_op);
//  return UnaryOp<arrow::BooleanBuilder, bool>(array, op, append_op);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
  std::cout <<  "cost"
     << double(duration) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
     << "s" << std::endl;
   return result;
}

std::shared_ptr<arrow::Array> ST_GeometryType(
    const std::shared_ptr<arrow::Array>& array) {

  auto append_op = [](arrow::StringBuilder& builder, ArrayValue<std::string>& item) {
  	builder.Append(item.item_value);
  };

  auto op = [](ArrayValue<std::string> &value, OGRGeometry* geo) {
    std::string name = std::string("ST_") + geo->getGeometryName();
    value.item_value = std::move(name);
    return false;
  };

  return UnaryOp<arrow::StringBuilder, std::string>(array, op, append_op);
}

std::shared_ptr<arrow::Array> ST_IsSimple(const std::shared_ptr<arrow::Array>& array) {
  const char* papszOptions[] = {(const char*)"ADD_INTERMEDIATE_POINT=YES", nullptr};

  auto append_op = [](arrow::BooleanBuilder& builder, ArrayValue<bool>& item) {
  	builder.Append(item.item_value);
  };

  auto op = [&papszOptions](ArrayValue<bool>& value,
                                           OGRGeometry* geo) {
    HasCircularVisitor has_circular;
    has_circular.reset();
    geo->accept(&has_circular);
    if (has_circular.has_circular()) {
      auto linear = geo->getLinearGeometry(0, papszOptions);
      value.item_value = linear->IsSimple() != 0;
      OGRGeometryFactory::destroyGeometry(linear);
    } else {
      value.item_value = geo->IsSimple() != 0;
    }
    return false;
  };

  return UnaryOp<arrow::BooleanBuilder, bool>(array, op, append_op);
}

std::shared_ptr<arrow::Array> ST_NPoints(const std::shared_ptr<arrow::Array>& array) {

  auto append_op = [](arrow::Int64Builder& builder, ArrayValue<int64_t>& item) {
    builder.Append(item.item_value);
  };

  auto op = [](ArrayValue<int64_t>& value, OGRGeometry* geo) {
    NPointsVisitor npoints;
    npoints.reset();
    geo->accept(&npoints);
    value.item_value = npoints.npoints();
    return false;
  };

  return UnaryOp<arrow::Int64Builder, int64_t>(array, op, append_op);
}

std::shared_ptr<arrow::Array> ST_Envelope(const std::shared_ptr<arrow::Array>& array) {

  auto append_op = [](arrow::BinaryBuilder& builder, 
		  ArrayValue<std::unique_ptr<OGRGeometry>> &item){
     AppendWkbNDR(builder, item.item_value.get());
  };

  auto op = [](ArrayValue<std::unique_ptr<OGRGeometry>> &item, OGRGeometry* geo) {
    OGREnvelope env;
    bool result = false;
    if (geo->IsEmpty()) {
      item.item_value = std::unique_ptr<OGRGeometry>(geo->clone());
      result = true;
    } else {
      result = false;
      OGR_G_GetEnvelope(geo, &env);
      if (env.MinX == env.MaxX) {    // vertical line or Point
        if (env.MinY == env.MaxY) {  // point
	  item.item_value = std::move(std::make_unique<OGRPoint>(env.MinX, env.MinY));
        } else {  // line
	  auto line = std::make_unique<OGRLineString>();
          line->addPoint(env.MinX, env.MinY);
          line->addPoint(env.MinX, env.MaxY);
	  item.item_value = std::move(line);
        }
      } else {
        if (env.MinY == env.MaxY) {  // horizontal line
	  auto line = std::make_unique<OGRLineString>();
          line->addPoint(env.MinX, env.MinY);
          line->addPoint(env.MaxX, env.MinY);
	  item.item_value = std::move(line);
        } else {  // polygon
          OGRLinearRing ring;
          ring.addPoint(env.MinX, env.MinY);
          ring.addPoint(env.MinX, env.MaxY);
          ring.addPoint(env.MaxX, env.MaxY);
          ring.addPoint(env.MaxX, env.MinY);
          ring.addPoint(env.MinX, env.MinY);
	  auto polygon = std::make_unique<OGRPolygon>();
          polygon->addRing(&ring);
	  item.item_value = std::move(polygon);
        }
      }
    }
    return result;
  };

  return UnaryOp<arrow::BinaryBuilder, std::unique_ptr<OGRGeometry>>(array, op, append_op);
}

/************************ GEOMETRY PROCESSING ************************/
std::vector<std::shared_ptr<arrow::Array>> ST_Buffer(
    const std::shared_ptr<arrow::Array>& array, double buffer_distance,
    int n_quadrant_segments) {

  auto append_op = [](ChunkArrayBuilder<arrow::BinaryBuilder>& builder, 
		  ArrayValue<OGRGeometry*> item){
	  auto result = AppendWkb(builder, item.item_value);
	  OGRGeometryFactory::destroyGeometry(item.item_value);
	  return result;
  };

  auto op = [&buffer_distance, &n_quadrant_segments](ArrayValue<OGRGeometry *> & item, OGRGeometry* geo) {
    auto buffer = geo->Buffer(buffer_distance, n_quadrant_segments);
    item.item_value = buffer;
    return false;
  };

  return UnaryOp<arrow::BinaryBuilder, OGRGeometry*>(array, op, append_op);
}

std::shared_ptr<arrow::Array> ST_PrecisionReduce(
  const std::shared_ptr<arrow::Array>& geometries, int32_t precision) {

  auto append_op = [](arrow::BinaryBuilder& builder, ArrayValue<OGRGeometry *> &item){
    AppendWkbNDR(builder, item.item_value);
    OGRGeometryFactory::destroyGeometry(item.item_value);
  };

  auto op = [&precision](ArrayValue<OGRGeometry*> & item, OGRGeometry* geo) {
    PrecisionReduceVisitor precision_reduce_visitor(precision);
    geo->accept(&precision_reduce_visitor);
    item.item_value = geo;
    return true;
  };

  return UnaryOp<arrow::BinaryBuilder, OGRGeometry *>(geometries, op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Intersection(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {

  std::vector<std::vector<std::shared_ptr<arrow::Array>>> array_list{geo1, geo2};
  std::vector<ChunkArrayIdx<WkbItem>> idx_list(2);

  ChunkArrayBuilder<arrow::BinaryBuilder> builder;

  std::vector<std::shared_ptr<arrow::Array>> result_array;

  auto has_curve = new HasCurveVisitor;
  OGRGeometryCollection empty;

  bool is_null = false;
  while (GetNextValue(array_list, idx_list, is_null)) {
    auto ogr1 = idx_list[0].item_value.ToGeometry();
    auto ogr2 = idx_list[1].item_value.ToGeometry();

    ogr1 = Wrapper_CurveToLine(ogr1, has_curve);
    ogr2 = Wrapper_CurveToLine(ogr2, has_curve);

    if ((ogr1 == nullptr) && (ogr2 == nullptr)) {
      builder.array_builder.AppendNull();
    } else if ((ogr1 == nullptr) || (ogr2 == nullptr)) {
      auto array_ptr = AppendWkb(builder, &empty);
      if (array_ptr != nullptr) result_array.push_back(array_ptr);
    } else {
      auto rst = ogr1->Intersection(ogr2);
      if (rst == nullptr) {
        builder.array_builder.AppendNull();
      } else if (rst->IsEmpty()) {
        auto array_ptr = AppendWkb(builder, &empty);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      } else {
        auto array_ptr = AppendWkb(builder, rst);
        if (array_ptr != nullptr) result_array.push_back(array_ptr);
      }
      OGRGeometryFactory::destroyGeometry(rst);
    }
    OGRGeometryFactory::destroyGeometry(ogr1);
    OGRGeometryFactory::destroyGeometry(ogr2);
  }

  delete has_curve;

  std::shared_ptr<arrow::Array> array_ptr;
  CHECK_ARROW(builder.array_builder.Finish(&array_ptr));
  result_array.push_back(array_ptr);
  return result_array;
}

std::shared_ptr<arrow::Array> ST_MakeValid(const std::shared_ptr<arrow::Array>& array) {
  auto wkb = std::static_pointer_cast<arrow::BinaryArray>(array);
  int len = wkb->length();
  arrow::BinaryBuilder builder;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkb(wkb, i);
    if (geo == nullptr) {
      builder.AppendNull();
    } else {
      if (geo->IsValid()) {
        arrow::BinaryArray::offset_type offset;
        auto data_ptr = wkb->GetValue(i, &offset);
        builder.Append(data_ptr, offset);
      } else {
        auto make_valid = geo->MakeValid();
        AppendWkbNDR(builder, make_valid);
        OGRGeometryFactory::destroyGeometry(make_valid);
      }
    }
    OGRGeometryFactory::destroyGeometry(geo);
  }
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_SimplifyPreserveTopology(
    const std::shared_ptr<arrow::Array>& array, double distance_tolerance) {

  auto append_op = [](arrow::BinaryBuilder& builder, ArrayValue<OGRGeometry*> &item){
    AppendWkbNDR(builder, item.item_value);
    OGRGeometryFactory::destroyGeometry(item.item_value);
  };

  auto op = [&distance_tolerance](ArrayValue<OGRGeometry*> &item, OGRGeometry* geo) {
    item.item_value = geo->SimplifyPreserveTopology(distance_tolerance);
    return false;
  };

  return UnaryOp<arrow::BinaryBuilder, OGRGeometry *>(array, op, append_op);
}

std::shared_ptr<arrow::Array> ST_Centroid(const std::shared_ptr<arrow::Array>& array) {

  auto append_op = [](arrow::BinaryBuilder& builder, ArrayValue<std::unique_ptr<OGRGeometry>> &item){
      AppendWkbNDR(builder, item.item_value.get());
  };

  auto op = [](ArrayValue<std::unique_ptr<OGRGeometry>> &item, OGRGeometry* geo) {
    auto centro_point = std::make_unique<OGRPoint>();
    auto err_code = geo->Centroid(centro_point.get());
    if (err_code == OGRERR_NONE) {
      item.item_value = std::move(centro_point);
    } else {
      item.is_null = true;
    }
    return false;
  };
  return UnaryOp<arrow::BinaryBuilder, std::unique_ptr<OGRGeometry>>(array, op, append_op);
}

std::shared_ptr<arrow::Array> ST_ConvexHull(const std::shared_ptr<arrow::Array>& array) {

  auto append_op = [](arrow::BinaryBuilder& builder, ArrayValue<OGRGeometry*> &item){
    AppendWkbNDR(builder, item.item_value);
    OGRGeometryFactory::destroyGeometry(item.item_value);
  };

  auto op = [](ArrayValue<OGRGeometry*> &item, OGRGeometry* geo) {
    item.item_value = geo->ConvexHull();
    return false;
  };
  return UnaryOp<arrow::BinaryBuilder, OGRGeometry*>(array, op, append_op);
}

/*
 * The detailed EPSG information can be found at EPSG.io [https://epsg.io/]
 */
std::shared_ptr<arrow::Array> ST_Transform(
    const std::shared_ptr<arrow::Array>& geometries, const std::string& src_rs,
    const std::string& dst_rs) {

  OGRSpatialReference oSrcSRS;
  oSrcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  if (oSrcSRS.SetFromUserInput(src_rs.c_str()) != OGRERR_NONE) {
    std::string err_msg = "faild to tranform with sourceCRS = " + src_rs;
    throw std::runtime_error(err_msg);
  }

  OGRSpatialReference oDstS;
  oDstS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  if (oDstS.SetFromUserInput(dst_rs.c_str()) != OGRERR_NONE) {
    std::string err_msg = "faild to tranform with targetCRS = " + dst_rs;
    throw std::runtime_error(err_msg);
  }

  void* poCT = OCTNewCoordinateTransformation(&oSrcSRS, &oDstS);


  auto append_op = [&poCT](arrow::BinaryBuilder& builder, ArrayValue<OGRGeometry*> &item) {
      AppendWkbNDR(builder, item.item_value);
      OGRGeometryFactory::destroyGeometry(item.item_value);
  };

  auto op = [&poCT](ArrayValue<OGRGeometry*> &item, OGRGeometry* geo) {
    auto err_code = geo->transform((OGRCoordinateTransformation*)poCT);
    if (err_code == OGRERR_NONE) {
      item.item_value = geo;
    } else {
      item.is_null = true;
    }
    return true;
  };
  auto results = UnaryOp<arrow::BinaryBuilder, OGRGeometry*>(geometries, op, append_op);
  OCTDestroyCoordinateTransformation(poCT);
  return results;
}

std::vector<std::shared_ptr<arrow::Array>> ST_CurveToLine(
    const std::shared_ptr<arrow::Array>& geometries) {

  auto append_op = [](ChunkArrayBuilder<arrow::BinaryBuilder>& builder, ArrayValue<OGRGeometry*> &item) {
    auto array_ptr = AppendWkb(builder, item.item_value);
    OGRGeometryFactory::destroyGeometry(item.item_value);
    return array_ptr;
  };

  auto op = [](ArrayValue<OGRGeometry*> &item, OGRGeometry* geo) {
    item.item_value = geo->getLinearGeometry();
    return false;
  };

  return UnaryOp<arrow::BinaryBuilder, OGRGeometry*>(geometries, op, append_op);
}

/************************ MEASUREMENT FUNCTIONS ************************/

std::shared_ptr<arrow::Array> ST_Area(const std::shared_ptr<arrow::Array>& geometries) {

  auto append_op = [](arrow::DoubleBuilder& builder, ArrayValue<double> &item) {
  	builder.Append(item.item_value);
  };

  auto op = [](ArrayValue<double> &item, OGRGeometry* geo) {
    AreaVisitor area;
    area.reset();
    geo->accept(&area);
    item.item_value = area.area();
    return false;
  };

  return UnaryOp<arrow::DoubleBuilder, double>(geometries, op, append_op);
}

std::shared_ptr<arrow::Array> ST_Length(const std::shared_ptr<arrow::Array>& geometries) {

  auto append_op = [](arrow::DoubleBuilder& builder, ArrayValue<double> &item) {
  	builder.Append(item.item_value);
  };

  auto op = [](ArrayValue<double> &item, OGRGeometry* geo) {
    LengthVisitor len_sum ;
    len_sum.reset();
    geo->accept(&len_sum);
    item.item_value = len_sum.length();
    return false;
  };

  return UnaryOp<arrow::DoubleBuilder, double>(geometries, op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_HausdorffDistance(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {


  auto append_op = [](ChunkArrayBuilder<arrow::DoubleBuilder>& builder, ArrayValue<double> &item){
      return AppendDouble(builder, item.item_value);
  };

  auto op = [](ArrayValue<double> &item, OGRGeometry* ogr1, OGRGeometry* ogr2) {

    auto geos_ctx = OGRGeometry::createGEOSContext();
    if (ogr1->IsEmpty() || ogr2->IsEmpty()) {
      item.is_null = true;
    } else {
      auto geos1 = ogr1->exportToGEOS(geos_ctx);
      auto geos2 = ogr2->exportToGEOS(geos_ctx);
      double dist;
      int geos_err = GEOSHausdorffDistance_r(geos_ctx, geos1, geos2, &dist);
      if (geos_err == 0) {  // geos error
        dist = -1;
      }
      GEOSGeom_destroy_r(geos_ctx, geos1);
      GEOSGeom_destroy_r(geos_ctx, geos2);
      item.item_value = dist;
    }
    OGRGeometry::freeGEOSContext(geos_ctx);
  };
  return BinaryOp<arrow::DoubleBuilder, double>(geo1, geo2, op, nullptr, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_DistanceSphere(
    const std::vector<std::shared_ptr<arrow::Array>>& point_left,
    const std::vector<std::shared_ptr<arrow::Array>>& point_right) {

  auto append_op = [](ChunkArrayBuilder<arrow::DoubleBuilder>& builder, ArrayValue<double> &item){
        return AppendDouble(builder, item.item_value);
  };

  auto distance = [](double fromlon, double fromlat, double tolon, double tolat) {
    double latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886;
    double longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886;
    double latitudeH = sin(latitudeArc * 0.5);
    latitudeH *= latitudeH;
    double lontitudeH = sin(longitudeArc * 0.5);
    lontitudeH *= lontitudeH;
    double tmp = cos(fromlat * 0.017453292519943295769236907684886) *
                 cos(tolat * 0.017453292519943295769236907684886);
    return 6372797.560856 * (2.0 * asin(sqrt(latitudeH + tmp * lontitudeH)));
  };

  auto op = [&distance](ArrayValue<double> & item, OGRGeometry* g1,
                        OGRGeometry* g2) {
    if ((g1->getGeometryType() != wkbPoint) || (g2->getGeometryType() != wkbPoint)) {
      item.is_null = true;
    } else {
      auto p1 = reinterpret_cast<OGRPoint*>(g1);
      auto p2 = reinterpret_cast<OGRPoint*>(g2);
      double fromlat = p1->getX();
      double fromlon = p1->getY();
      double tolat = p2->getX();
      double tolon = p2->getY();
      if ((fromlat > 180) || (fromlat < -180) || (fromlon > 90) || (fromlon < -90) ||
          (tolat > 180) || (tolat < -180) || (tolon > 90) || (tolon < -90)) {
        item.is_null = true;
      } else {
	item.item_value = distance(fromlat, fromlon, tolat, tolon);
      }
    }
  };

  return BinaryOp<arrow::DoubleBuilder, double>(point_left, point_right, op, nullptr, append_op);
}


std::vector<std::shared_ptr<arrow::Array>> ST_Distance(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {

  auto append_op = [](ChunkArrayBuilder<arrow::DoubleBuilder>& builder, ArrayValue<double> &item){
        return  AppendDouble(builder, item.item_value);
  };

  auto op = [](ArrayValue<double> &item, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    if (ogr1->IsEmpty() || ogr2->IsEmpty()) {
      item.is_null = true;
    } else {
      auto dist = ogr1->Distance(ogr2);
      if (dist < 0) {
        item.is_null = true;
      } else {
        item.item_value = dist;
      }
    }
  };

  return BinaryOp<arrow::DoubleBuilder, double>(geo1, geo2, op, nullptr, append_op);
}

/************************ SPATIAL RELATIONSHIP ************************/

/*************************************************
 * https://postgis.net/docs/ST_Equals.html
 * Returns TRUE if the given Geometries are "spatially equal".
 * Use this for a 'better' answer than '='.
 * Note by spatially equal we mean ST_Within(A,B) = true and ST_Within(B,A) = true and
 * also mean ordering of points can be different but represent the same geometry
 * structure. To verify the order of points is consistent, use ST_OrderingEquals (it must
 * be noted ST_OrderingEquals is a little more stringent than simply verifying order of
 * points are the same).
 * ***********************************************/

std::vector<std::shared_ptr<arrow::Array>> ST_Equals(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {

  auto append_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, ArrayValue<bool> &item){
      return AppendBoolean(builder, item.item_value);
  };

  auto op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    if (ogr1->IsEmpty() && ogr2->IsEmpty()) {
      item.item_value = true;
    } else if (ogr1->Within(ogr2) && ogr2->Within(ogr1)) {
      item.item_value = true;
    } else {
      item.item_value = false;
    }
  };

  auto null_op = [](ArrayValue<bool> & item, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { item.item_value = false; };

  return BinaryOp<arrow::BooleanBuilder, bool>(geo1, geo2, op, null_op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Touches(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {

  auto append_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, ArrayValue<bool> &item){
    return AppendBoolean(builder, item.item_value);
  };

  auto op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
	  item.item_value = ogr1->Touches(ogr2) != 0;
  };

  auto null_op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { item.item_value = false; };

  return BinaryOp<arrow::BooleanBuilder, bool>(geo1, geo2, op, null_op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Overlaps(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {

  auto append_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, ArrayValue<bool> &item){
    return AppendBoolean(builder, item.item_value);
  };

  auto op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    item.item_value = ogr1->Overlaps(ogr2) != 0;
  };

  auto null_op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { item.item_value = false; };

  return BinaryOp<arrow::BooleanBuilder, bool>(geo1, geo2, op, null_op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Crosses(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {

  auto append_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, ArrayValue<bool> &item){
    return AppendBoolean(builder, item.item_value);
  };

  auto op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    item.item_value = ogr1->Crosses(ogr2) != 0;
  };

  auto null_op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { item.item_value = false; };

  return BinaryOp<arrow::BooleanBuilder, bool>(geo1, geo2, op, null_op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Contains(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {

  auto append_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, ArrayValue<bool> &item){
    return AppendBoolean(builder, item.item_value);
  };

  auto op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    item.item_value = ogr1->Contains(ogr2) != 0;
  };

  auto null_op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { item.item_value = false; };

  return BinaryOp<arrow::BooleanBuilder, bool>(geo1, geo2, op, null_op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Intersects(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {

  auto append_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, ArrayValue<bool> &item){
    return AppendBoolean(builder, item.item_value);
  };

  auto op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    item.item_value = ogr1->Intersects(ogr2) != 0;
  };

  auto null_op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { item.item_value = false; };

  return BinaryOp<arrow::BooleanBuilder, bool>(geo1, geo2, op, null_op, append_op);
}

std::vector<std::shared_ptr<arrow::Array>> ST_Within(
    const std::vector<std::shared_ptr<arrow::Array>>& geo1,
    const std::vector<std::shared_ptr<arrow::Array>>& geo2) {
  
  auto append_op = [](ChunkArrayBuilder<arrow::BooleanBuilder>& builder, ArrayValue<bool> &item){
      return AppendBoolean(builder, item.item_value);
  };

  auto op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
               OGRGeometry* ogr2) {
    bool flag = true;
    bool result = false;
    do {
      /*
       * speed up for point within circle
       * point pattern : 'POINT ( x y )'
       * circle pattern : 'CurvePolygon ( CircularString ( x1 y1, x2 y2, x1 y2 ) )'
       *                   if the circularstring has 3 points and closed,
       *                   it becomes a circle,
       *                   the centre is (x1+x2)/2, (y1+y2)/2
       *                   the radius is sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y2-y2))/2
       */
      auto type1 = ogr1->getGeometryType();
      if (type1 != wkbPoint) break;
      auto point = reinterpret_cast<OGRPoint*>(ogr1);

      auto type2 = ogr2->getGeometryType();
      if (type2 != wkbCurvePolygon) break;
      auto curve_poly = reinterpret_cast<OGRCurvePolygon*>(ogr2);

      auto curve_it = curve_poly->begin();
      if (curve_it == curve_poly->end()) break;
      auto curve = *curve_it;
      ++curve_it;
      if (curve_it != curve_poly->end()) break;

      auto curve_type = curve->getGeometryType();
      if (curve_type != wkbCircularString) break;
      auto circular_string = reinterpret_cast<OGRCircularString*>(curve);
      if (circular_string->getNumPoints() != 3) break;
      if (!circular_string->get_IsClosed()) break;

      auto circular_point_it = circular_string->begin();
      auto circular_point = &(*circular_point_it);
      if (circular_point->getGeometryType() != wkbPoint) break;
      auto p0_x = circular_point->getX();
      auto p0_y = circular_point->getY();

      ++circular_point_it;
      circular_point = &(*circular_point_it);
      if (circular_point->getGeometryType() != wkbPoint) break;
      auto p1_x = circular_point->getX();
      auto p1_y = circular_point->getY();

      auto d_x = (p0_x + p1_x) / 2 - point->getX();
      auto d_y = (p0_y + p1_y) / 2 - point->getY();
      auto dd = 4 * (d_x * d_x + d_y * d_y);
      auto l_x = p0_x - p1_x;
      auto l_y = p0_y - p1_y;
      auto ll = l_x * l_x + l_y * l_y;
      result = dd <= ll;

      flag = false;
    } while (0);

    if (flag) result = ogr1->Within(ogr2) != 0;

    item.item_value = result;
  };

  auto null_op = [](ArrayValue<bool> &item, OGRGeometry* ogr1,
                    OGRGeometry* ogr2) { item.item_value = false; };

  return BinaryOp<arrow::BooleanBuilder, bool>(geo1, geo2, op, null_op, append_op);
}

/*********************** AGGREGATE FUNCTIONS ***************************/

std::shared_ptr<arrow::Array> ST_Union_Aggr(const std::shared_ptr<arrow::Array>& geo) {
  auto len = geo->length();
  auto wkt = std::static_pointer_cast<arrow::BinaryArray>(geo);
  std::vector<OGRGeometry*> union_agg;
  OGRPolygon empty_polygon;
  OGRGeometry *g0, *g1;
  OGRGeometry *u0, *u1;
  auto has_curve = new HasCurveVisitor;
  for (int i = 0; i <= len / 2; i++) {
    if ((i * 2) < len) {
      g0 = Wrapper_createFromWkb(wkt, 2 * i);
      g0 = Wrapper_CurveToLine(g0, has_curve);
    } else {
      g0 = nullptr;
    }

    if ((i * 2 + 1) < len) {
      g1 = Wrapper_createFromWkb(wkt, 2 * i + 1);
      g1 = Wrapper_CurveToLine(g1, has_curve);
    } else {
      g1 = nullptr;
    }

    if (g0 != nullptr) {
      auto type = wkbFlatten(g0->getGeometryType());
      if (type == wkbMultiPolygon) {
        u0 = g0->UnionCascaded();
        OGRGeometryFactory::destroyGeometry(g0);
      } else {
        u0 = g0;
      }
    } else {
      u0 = nullptr;
    }

    if (g1 != nullptr) {
      auto type = wkbFlatten(g1->getGeometryType());
      if (type == wkbMultiPolygon) {
        u1 = g1->UnionCascaded();
        OGRGeometryFactory::destroyGeometry(g1);
      } else {
        u1 = g1;
      }
    } else {
      u1 = nullptr;
    }

    if ((u0 != nullptr) && (u1 != nullptr)) {
      OGRGeometry* ua = u0->Union(u1);
      union_agg.push_back(ua);
      OGRGeometryFactory::destroyGeometry(u0);
      OGRGeometryFactory::destroyGeometry(u1);
    } else if ((u0 != nullptr) && (u1 == nullptr)) {
      union_agg.push_back(u0);
    } else if ((u0 == nullptr) && (u1 != nullptr)) {
      union_agg.push_back(u1);
    }
  }
  len = union_agg.size();
  while (len > 1) {
    std::vector<OGRGeometry*> union_tmp;
    for (int i = 0; i <= len / 2; ++i) {
      if (i * 2 < len) {
        u0 = union_agg[i * 2];
      } else {
        u0 = nullptr;
      }

      if (i * 2 + 1 < len) {
        u1 = union_agg[i * 2 + 1];
      } else {
        u1 = nullptr;
      }

      if ((u0 != nullptr) && (u1 != nullptr)) {
        OGRGeometry* ua = u0->Union(u1);
        union_tmp.push_back(ua);
        OGRGeometryFactory::destroyGeometry(u0);
        OGRGeometryFactory::destroyGeometry(u1);
      } else if ((u0 != nullptr) && (u1 == nullptr)) {
        union_tmp.push_back(u0);
      } else if ((u0 == nullptr) && (u1 != nullptr)) {
        union_tmp.push_back(u1);
      }
    }
    union_agg = std::move(union_tmp);
    len = union_agg.size();
  }
  arrow::BinaryBuilder builder;
  if (union_agg.empty()) {
    builder.AppendNull();
  } else {
    AppendWkbNDR(builder, union_agg[0]);
    OGRGeometryFactory::destroyGeometry(union_agg[0]);
  }
  delete has_curve;
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

std::shared_ptr<arrow::Array> ST_Envelope_Aggr(
    const std::shared_ptr<arrow::Array>& geometries) {
  auto wkt_geometries = std::static_pointer_cast<arrow::BinaryArray>(geometries);
  auto len = geometries->length();
  double inf = std::numeric_limits<double>::infinity();
  double xmin = inf;
  double xmax = -inf;
  double ymin = inf;
  double ymax = -inf;

  OGREnvelope env;
  bool set_env = false;
  for (int i = 0; i < len; ++i) {
    auto geo = Wrapper_createFromWkb(wkt_geometries, i);
    if (geo == nullptr) continue;
    if (geo->IsEmpty()) continue;
    set_env = true;
    OGR_G_GetEnvelope(geo, &env);
    if (env.MinX < xmin) xmin = env.MinX;
    if (env.MaxX > xmax) xmax = env.MaxX;
    if (env.MinY < ymin) ymin = env.MinY;
    if (env.MaxY > ymax) ymax = env.MaxY;
    OGRGeometryFactory::destroyGeometry(geo);
  }
  arrow::BinaryBuilder builder;
  OGRPolygon polygon;
  if (set_env) {
    OGRLinearRing ring;
    ring.addPoint(xmin, ymin);
    ring.addPoint(xmin, ymax);
    ring.addPoint(xmax, ymax);
    ring.addPoint(xmax, ymin);
    ring.addPoint(xmin, ymin);
    polygon.addRing(&ring);
  }
  AppendWkbNDR(builder, &polygon);
  std::shared_ptr<arrow::Array> results;
  CHECK_ARROW(builder.Finish(&results));
  return results;
}

void set_parallelism(int parallelism){
    if (parallelism >= 0){
	omp_parallelism = parallelism;
    }
}

int get_parallelism(){
    if(omp_parallelism){
	return omp_parallelism;
    }
    if (0 == num_of_procs){
	num_of_procs = omp_get_num_procs();
    }
    return num_of_procs;
}


}  // namespace gdal
}  // namespace gis
}  // namespace arctern
