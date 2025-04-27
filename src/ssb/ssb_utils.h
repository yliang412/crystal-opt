#include <fstream>
#include <iostream>
#include <string>

using namespace std;

/** @brief Defines the scale factor here. */
#define SF 128

//#define BASE_PATH "/afs/cs.cmu.edu/user/yuchenl3/scs-workspace/crystal-opt/test/ssb/data/"
#define BASE_PATH "/root/15740/crystal-opt/test/ssb/data/"

#if SF == 1
#define DATA_DIR BASE_PATH "s1_columnar/"
#define LO_LEN 6001171
#define P_LEN 200000
#define S_LEN 2000
#define C_LEN 30000
#define D_LEN 2556
#elif SF == 2
#define DATA_DIR BASE_PATH "s2_columnar/"
#define LO_LEN 11998051
#define P_LEN 400000
#define S_LEN 4000
#define C_LEN 60000
#define D_LEN 2556
#elif SF == 4
#define DATA_DIR BASE_PATH "s4_columnar/"
#define LO_LEN 23996670
#define P_LEN 600000
#define S_LEN 8000
#define C_LEN 120000
#define D_LEN 2556
#elif SF == 8
#define DATA_DIR BASE_PATH "s8_columnar/"
#define LO_LEN 47989129
#define P_LEN 800000
#define S_LEN 16000
#define C_LEN 240000
#define D_LEN 2556
#elif SF == 16
#define DATA_DIR BASE_PATH "s16_columnar/"
#define LO_LEN 95988758
#define P_LEN 1000000
#define S_LEN 32000
#define C_LEN 480000
#define D_LEN 2556
#elif SF == 32
#define DATA_DIR BASE_PATH "s32_columnar/"
#define LO_LEN 192000754
#define P_LEN 1200000
#define S_LEN 64000
#define C_LEN 960000
#define D_LEN 2556
#elif SF == 64
#define DATA_DIR BASE_PATH "s64_columnar/"
#define LO_LEN 384016864
#define P_LEN 1400000
#define S_LEN 128000
#define C_LEN 1920000
#define D_LEN 2556
#elif SF == 128
#define DATA_DIR BASE_PATH "s128_columnar/"
#define LO_LEN 768047048
#define P_LEN 1600000
#define S_LEN 256000
#define C_LEN 3840000
#define D_LEN 2556
#elif SF == 256
#define DATA_DIR BASE_PATH "s256_columnar/"
#define LO_LEN 1536094096
#define P_LEN 1800000
#define S_LEN 512000
#define C_LEN 7680000
#define D_LEN 2556
#else // 20
#define DATA_DIR BASE_PATH "s20_columnar/"
#define LO_LEN 119994746
#define P_LEN 1000000
#define S_LEN 40000
#define C_LEN 600000
#define D_LEN 2556
#endif

uint32_t roundUpToPowerOfTwo(uint32_t n) {
  if (n == 0)
      return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

int index_of(string *arr, int len, string val) {
  for (int i = 0; i < len; i++)
    if (arr[i] == val)
      return i;

  return -1;
}

string lookup(string col_name) {
  string lineorder[] = {"lo_orderkey",      "lo_linenumber",    "lo_custkey",
                        "lo_partkey",       "lo_suppkey",       "lo_orderdate",
                        "lo_orderpriority", "lo_shippriority",  "lo_quantity",
                        "lo_extendedprice", "lo_ordtotalprice", "lo_discount",
                        "lo_revenue",       "lo_supplycost",    "lo_tax",
                        "lo_commitdate",    "lo_shipmode"};
  string part[] = {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
                   "p_color",   "p_type", "p_size", "p_container"};
  string supplier[] = {"s_suppkey", "s_name",   "s_address", "s_city",
                       "s_nation",  "s_region", "s_phone"};
  string customer[] = {"c_custkey", "c_name",   "c_address", "c_city",
                       "c_nation",  "c_region", "c_phone",   "c_mktsegment"};
  string date[] = {"d_datekey",
                   "d_date",
                   "d_dayofweek",
                   "d_month",
                   "d_year",
                   "d_yearmonthnum",
                   "d_yearmonth",
                   "d_daynuminweek",
                   "d_daynuminmonth",
                   "d_daynuminyear",
                   "d_sellingseason",
                   "d_lastdayinweekfl",
                   "d_lastdayinmonthfl",
                   "d_holidayfl",
                   "d_weekdayfl"};

  if (col_name[0] == 'l') {
    int index = index_of(lineorder, 17, col_name);
    return "LINEORDER" + to_string(index);
  } else if (col_name[0] == 's') {
    int index = index_of(supplier, 7, col_name);
    return "SUPPLIER" + to_string(index);
  } else if (col_name[0] == 'c') {
    int index = index_of(customer, 8, col_name);
    return "CUSTOMER" + to_string(index);
  } else if (col_name[0] == 'p') {
    int index = index_of(part, 9, col_name);
    return "PART" + to_string(index);
  } else if (col_name[0] == 'd') {
    int index = index_of(date, 15, col_name);
    return "DDATE" + to_string(index);
  }

  return "";
}

/**
  * @brief Loads a column from the data directory.
  * @param col_name The name of the column to load.
  * @param num_entries The number of entries in the column.
  * @return A pointer to the loaded column. NULL on failure.
  */
template <typename T> T *loadColumn(string col_name, int num_entries) {
  T *h_col = new T[num_entries];
  string filename = DATA_DIR + lookup(col_name);
  ifstream colData(filename.c_str(), ios::in | ios::binary);
  if (!colData) {
    return NULL;
  }

  colData.read((char *)h_col, num_entries * sizeof(T));
  return h_col;
}

/**
  * @brief Stores a column to the data directory.
  * @param col_name The name of the column to store.
  * @param num_entries The number of entries in the column.
  * @param h_col The column to store.
  * @return 0 on success, -1 on failure.
  */
template <typename T>
int storeColumn(string col_name, int num_entries, int *h_col) {
  string filename = DATA_DIR + lookup(col_name);
  ofstream colData(filename.c_str(), ios::out | ios::binary);
  if (!colData) {
    return -1;
  }

  colData.write((char *)h_col, num_entries * sizeof(T));
  return 0;
}
