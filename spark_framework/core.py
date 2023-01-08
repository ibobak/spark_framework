import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis, spearmanr, pearsonr

from datetime import datetime
import re

import psycopg2
from pgcopy import CopyManager

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StringType, Row, StructField, StructType, TimestampType, ByteType, ShortType, \
    IntegerType, LongType, FloatType, DoubleType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.storagelevel import StorageLevel
from pyspark.sql import Window
import pyspark.sql.functions as sf

from typing import TypeVar, Union, Tuple


ListOrStr = TypeVar('ListOrStr', list, str)


# region Module Initialization code - CALL init(globals()) FROM OUTSIDE BEFORE USING THE MODULE!

_spark: Union[SparkSession, None] = None


def init(a_spark: SparkSession):
    global _spark
    _spark = a_spark

# endregion


# region Constants

# connection info
C_TYPE = "type"
C_DFS = "dfs"
C_LOCAL = "local"
C_JDBC = "jdbc"
C_URL = "url"
C_SUPPORTED_CONN_TYPES = [C_DFS, C_LOCAL, C_JDBC]

# mandatory options for each connection type (at this moment just "url" is mandatory, this can be changed)
C_MUST_HAVE_OPTIONS = {C_DFS: [C_URL], C_LOCAL: [C_URL], C_JDBC: [C_URL]}

# hadoop/spark file formats
C_CSV = "csv"
C_CSV_GZ = "csv.gz"
C_PARQUET = "parquet"
C_AVRO = "avro"
C_DELTA = "delta"
C_JSON = "json"
C_JSON_GZ = "json.gz"
C_GZIP_CODEC = "org.apache.hadoop.io.compress.GzipCodec"
# mapping between format name and spark.read.format(VALUE)
C_FORMAT_MAP = {
    C_CSV_GZ: "csv",
    C_JSON_GZ: "json",
    C_AVRO: "com.databricks.spark.avro"
}

# Supported formats for reading/writing to file systems.
# If you change this value, make sure that you've done the changes in the load_df and save_df to create the appropriate readers
C_FS_SUPPORTED_FORMATS = [C_CSV, C_CSV_GZ, C_PARQUET, C_AVRO, C_DELTA, C_JSON, C_JSON_GZ]

# database connection parameters
C_HOST = "host"
C_DATABASE = "database"
C_DBNAME = "dbname"
C_PORT = "port"
C_DRIVER = "driver"
# parsing PostgreSQL connection string
C_PG_URL_PARSER = re.compile(r"jdbc:postgresql://([a-zA-Z0-9.-]+)(:(\d+))(/([a-zA-Z0-9.-]+)).*")

C_TEST_PASSED = "unit test passed"
C_TEST_FAILURE = "Execution terminated due to unit test failure: "

# endregion


# region Utility functions

def print_verbose(a_level, a_max_level, a_message):
    if a_level <= a_max_level:
        print(a_message)


def check_keys(a_keys: list, a_dict: dict):
    for k in a_keys:
        if k not in a_dict:
            return False
    return True


def join_path_char(a_join_char, *a_parts):
    result = []
    for index in range(0, len(a_parts)):
        s = a_parts[index]
        if index == 0:
            result.append(s)
        else:
            is_last_slash = result[len(result) - 1].endswith(a_join_char)
            is_this_slash = s.startswith(a_join_char)
            if not is_last_slash and not is_this_slash:
                result.append(a_join_char)
                result.append(s)
            elif is_last_slash and is_this_slash:
                if len(s) > 1:
                    result.append(s[1:])
            else:
                result.append(s)
    return "".join(result)


def join_path_generic(a_prefix, a_list_or_string: Union[list, str]):
    if isinstance(a_list_or_string, str):
        return join_path_char("/", a_prefix, a_list_or_string)
    return [join_path_char("/", a_prefix, item) for item in a_list_or_string]


def join_path_os(a_prefix, a_list_or_string: Union[list, str]):
    if isinstance(a_list_or_string, str):
        return join_path_char(os.sep, a_prefix, a_list_or_string)
    return [join_path_char(os.sep, a_prefix, item) for item in a_list_or_string]


def date_to_ymd_hms(a_date):
    return a_date.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def isnotebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules


def display_pdf(a_object):
    if isnotebook():
        from IPython.core.display import display as ipython_display
        ipython_display(a_object)
    else:
        print(a_object)


def seconds_passed(a_since_datetime, a_return_now=False):
    dtn = datetime.now()
    d = dtn - a_since_datetime
    result = d.seconds + d.microseconds / 1000000
    if a_return_now:
        return result, dtn
    return result

# endregion


# region Load/Save

def parse_conn_info(a_conn_info: dict):
    """Parses connection information.

    :param a_conn_info: dictionary with connection parameters. Mandatory are "type" and "url".

    :return: tuple (type, url, other parameters)
    """
    if C_TYPE not in a_conn_info:
        raise ValueError(f"You need to specity 'type' with one of the following values: {C_SUPPORTED_CONN_TYPES}")

    result_options = a_conn_info.copy()

    # pull the type
    result_type = result_options.pop(C_TYPE)
    if result_type not in C_SUPPORTED_CONN_TYPES:
        raise ValueError(f"The following connection types are supported: {C_SUPPORTED_CONN_TYPES}")

    # check if necessary options were passed
    if not check_keys(C_MUST_HAVE_OPTIONS[result_type], result_options):
        raise ValueError(f"You need to specify all these options for {result_type}: {C_MUST_HAVE_OPTIONS[result_type]}")

    # pull the url
    result_url = result_options.pop(C_URL)

    return result_type, result_url, result_options


def get_fs_reader(a_format, a_header, a_schema, a_verbose_level=3, **a_options):
    """Creates a reader object for specified parameters - format, header, schema, etc.

    :param a_format: data format. Must be one of C_FS_SUPPORTED_FORMATS
    :param a_header: True/False for csv and csv.gz indicating that there is a header
    :param a_schema: optional schema for csv and csv.gz file. If not set, it will be inferred.
    :param a_verbose_level: print verbosity level
    :param a_options: other options that should go to spark.read.option(k, v)

    :return: dataframe reader
    """
    # check input parametersd
    if a_format not in C_FS_SUPPORTED_FORMATS:
        raise ValueError(f"a_format must be one of the following: {C_FS_SUPPORTED_FORMATS}")
    if a_format in [C_CSV_GZ, C_CSV]:
        if not isinstance(a_header, bool):
            raise ValueError(f"For formats {C_CSV_GZ} or {C_CSV} you need to set a_header=True/False explicitly")
    else:
        # for non-csv format there should be no header or schema
        if isinstance(a_header, bool):
            raise ValueError(f"You should not set a_header for {a_format} format")

    # construct the result
    fmt = C_FORMAT_MAP.get(a_format, a_format)
    res = _spark.read.format(fmt)
    print_verbose(4, a_verbose_level, f"format: {fmt}")

    # for csv.gz apply the codec
    if a_format in {C_CSV_GZ, C_JSON_GZ}:
        print_verbose(4, a_verbose_level, f"option codec={C_GZIP_CODEC}")
        res = res.option("codec", C_GZIP_CODEC)

    if a_header:
        print_verbose(4, a_verbose_level, "option header=True")
        res = res.option("header", True)

    if a_schema:
        print_verbose(4, a_verbose_level, f"schema:\n{schema_to_str(a_schema)}")
        res = res.schema(a_schema)
    else:
        print_verbose(4, a_verbose_level, f"no schema specified, inferring")

    # other options
    if a_options:
        print_verbose(4, a_verbose_level, f"other options: {hide_sensitive_str(a_options)}")
        for k, v in a_options.items():
            res = res.option(k, v)

    print_verbose(4, a_verbose_level, "done creating the reader, returning it")
    return res


def get_jdbc_reader(a_table, a_query, a_table_options, a_url, a_verbose_level=3, **a_options):
    """Construct Spark JDBC reader.

    :param a_table: table name in the relational database
    :param a_url: JDBC connection string
    :param a_verbose_level: print verbosity level
    :param a_options: other options that should go to spark.read.option(k, v)

    :return:  dataframe reader
    """
    res = _spark.read.format(C_JDBC)
    
    if a_table is not None:
        print_verbose(4, a_verbose_level, f"dbtable: {a_table}")
        res = res.option("dbtable", a_table)
    if a_query is not None:    
        print_verbose(4, a_verbose_level, f"query: {a_query}")
        res = res.option("query", a_query)

    print_verbose(4, a_verbose_level, f"url: {a_url}")
    res = res.option('url', a_url)

    if a_table_options:
        print_verbose(4, a_verbose_level, f"other options: {hide_sensitive_str(a_table_options)}")
        for k, v in a_table_options.items():
            res = res.option(k, v)
            
    if a_options:
        print_verbose(4, a_verbose_level, f"other options: {hide_sensitive_str(a_options)}")
        for k, v in a_options.items():
            res = res.option(k, v)

    return res


stat_path = []


def load(a_souce_conn_info: dict, a_table,
         a_format=None, a_header=None, a_schema=None,
         a_cache=False, a_row_count=False, a_return_count=False,
         a_select=None, a_where=None, a_rename=None,
         a_show=False, a_show_limit=100, a_show_order_by=None,
         a_verbose_level=3, a_ps=False, a_temp_table=None):
    """Loads a dataframe

    :param a_souce_conn_info: source connection info. It must be a dictionary with just "type" and "url" mandatory parameters,
            the rest are optional.
    :param a_table: table name to read. For type=JDBC this is a table name,
            for file systems this is a path to folder/file or file mask relatively to the root dir (specified in the 'url')
            # a_table can be either string (in this case you need to explicitly set a_format, a_header, a_schema), or
            # a dictionary with keys "name", "format", "header", "schema"
    :param a_format: must be obe of these types: C_FS_SUPPORTED_FORMATS
    :param a_header: True/False for csv and csv.gz indicating that there is a header
    :param a_schema: optional schema for csv and csv.gz file. If not set, it will be inferred.
    :param a_cache: optional caching of dataframe after loading.
    :param a_row_count: count the rows of dataframe after loading.
            If a_cache and  a_row_count are both true, the dataframe will be immediately cached.
    :param a_return_count: returns count of rows in a tuple with the dataframe.
    :param a_verbose_level: print verbosity level

    :return: either a dataframe or a tuple (dataframe, count) if a_return_count==True
    """
    # in the case if the table name is a dict - get the format, header and schema from there, and check
    # if they are NOT set in the regular parameters
    a_query, a_table_options = None, {}
    if isinstance(a_show, bool) and a_show:
        a_show = a_show_limit
    if isinstance(a_table, dict):
        if a_format is not None or a_header is not None or a_schema is not None:
            raise ValueError("If you specify the a_table to be a dictionary of values, then you should NOT specify "
                             "a_format, a_header, a_schema separately because they should be specified in the table.")
        a_format = a_table.get("format", None)
        a_header = a_table.get("header", None)
        a_schema = a_table.get("schema", None)
        t = a_table
        a_table = t.get("name", None)
        a_query = t.get("query", None)
        if a_table is None and a_query is None:
            raise ValueError("Either 'name' or 'query' must be specified for the table")
        a_table_options = t.get("options", {})
        if a_table_options is not None and not isinstance(a_table_options, dict):
            raise ValueError("Table options must be either dict or None")
    elif not isinstance(a_table, str) and not isinstance(a_table, list):
        raise ValueError("a_table must be either dict(name, format, header, schema) or string or list")

    if a_table is None and a_query is None:
        raise ValueError("The a_table or a_query must be set")

    # parse the connection info
    source_type, source_url, source_options = parse_conn_info(a_souce_conn_info)

    if source_type == C_DFS or source_type == C_LOCAL:
        # construct the full path: for DFS use just /, for local filesystem use OS specific slash
        path = join_path_generic(source_url, a_table) if source_type == C_DFS else join_path_os(source_url, a_table)
        print_verbose(1, a_verbose_level, f"loading dataframe from {source_type} path {path}")
        stat_path.append((datetime.now(), path))
        result = get_fs_reader(a_format, a_header, a_schema, a_verbose_level=a_verbose_level, **a_table_options).load(path)
    elif source_type == C_JDBC:
        print_verbose(1, a_verbose_level, f"loading dataframe from table {a_table} at {source_url}")
        result = get_jdbc_reader(a_table, a_query, a_table_options, source_url, a_verbose_level=a_verbose_level, **source_options).load()
    else:
        raise NotImplementedError(f"Type {source_type} is not implemented yet")

    if a_select or a_where or a_rename:
        result = change(result, a_where=a_where, a_select=a_select, a_rename=a_rename)

    if a_cache:
        print_verbose(1, a_verbose_level, "caching dataframe")
        cache(result)

    cnt = -1
    if a_row_count or a_return_count:
        print_verbose(1, a_verbose_level, "counting rows")
        cnt = result.count()
        print_verbose(1, a_verbose_level, f"RC={cnt:,}")

    if a_ps:
        result.printSchema()

    if a_temp_table:
        result.createOrReplaceTempView(a_temp_table)
        print_verbose(1, a_verbose_level, f"Temp table created: {a_temp_table}")

    print_verbose(1, a_verbose_level, "loading done.")
    if a_show:
        result_to_show = result
        if a_show_order_by:
            result_to_show = change(result_to_show, a_order_by=a_show_order_by, a_verbose_level=0)
        show(result_to_show, a_limit=a_show, a_verbose_level=a_verbose_level)
    if a_return_count:
        return result, cnt
    return result


def pg_make_copy_to_func(a_columns, a_string_fields, a_dest_table, **a_pg_conn_params):
    """Prepares a function for copying partition into Postgres table using its native 'copy' method (fast loading).

    :param a_columns: list of columns to take from the input rows
    :param a_string_fields:  list of fields of string type:  they must be encoded as bytes() with 'utf-8'
    :param a_dest_table: destination table name. It must exist on the server before writing there.
    :param a_pg_conn_params: connection parameters
    :return: a function that opens a connection (using psycopg), uses copy manager to copy data, and closes the connection.

    List of additional readings:

        PostgreSQL adapter for Python, implements DB API 2.0:  http://initd.org/psycopg/
        python library for nice wrapping of Postgres copy routine into low level calls to psycopg2: https://pypi.org/project/pgcopy/
        foreachPartition: http://spark.apache.org/docs/latest/api/python/pyspark.sql.html
        Postgres copy operation: https://www.postgresql.org/docs/9.2/sql-copy.html
        Explanations of why we need renaming of columns: https://stackoverflow.com/questions/21796446/postgres-case-sensitivity
        Explanation of client_encoding: read 23.3.3. Automatic Character Set Conversion Between Server and Client: https://www.postgresql.org/docs/current/multibyte.html
        C library on top of which psycopg is working: https://www.postgresql.org/docs/current/libpq.html
        DB API 2.0 specification: https://www.python.org/dev/peps/pep-0249
    """
    def result_func(a_rows):
        a_pg_conn_params["client_encoding"] = 'utf-8'
        conn = psycopg2.connect(**a_pg_conn_params)
        mgr = CopyManager(conn, a_dest_table, a_columns)
        # this is done to speed up the process: in the case if there are no string fields, we don't have to
        # perform a check "if this is a string field then encode it" - iterator will work faster
        if len(a_string_fields) > 0:
            mgr.copy(tuple(r[c].encode('utf-8') if c in a_string_fields and r[c] else r[c] for c in a_columns) for r in a_rows)
        else:
            mgr.copy(tuple(r[c] for c in a_columns) for r in a_rows)
        conn.commit()
        conn.close()

    return result_func


def parse_pg_url(a_url):
    """Conversion of JDBC connection string into three separate parameters: host, port and database.
    It is necessary to do it in order to have a single format of the conneection info as "JDBC format".

    :param a_url: a valid JDBC connection string
    :return:  tuple (host, port and database)
    """
    m = C_PG_URL_PARSER.search(a_url)
    if m is None:
        raise ValueError("url must be in this format: jdbc:postgresql:/host[:port]/db")
    return m.group(1), m.group(3), m.group(5)


def pg_copy(a_df: DataFrame, a_dest_table: str, a_overwrite=True, a_verbose_level=3, **a_postgres_conn_params2):
    """Copies a dataframe into Postgres table using its 'copy' method.

    :param a_df: the dataframe
    :param a_dest_table: table name in Postgres
    :param a_overwrite: if True, the table will be truncated before writing, otherwise data will be appended
    :param a_verbose_level: printing verbosity level
    :param a_postgres_conn_params: there are two options:
            a) either pass "host", "database" and optional "port" or
            b) pass "url" in format of JDBC, e.g. jdbc:postgresql://glo-mda-psql-a3d.postgres.database.azure.com:5432/hmdapsqla3d
            In the "b" case we will parse host, port and database from the url.
    """
    conn_params = a_postgres_conn_params2.copy()

    # if we set the "url" (this is what will happen usually, because JDBC format is used to set up connection info),
    # then we need to the the host, port and database out of it.
    if C_URL in conn_params:
        if C_HOST in conn_params or C_PORT in conn_params or C_DATABASE in conn_params or C_DBNAME in conn_params:
            raise ValueError(f"If you set the url, then there should be no {C_HOST}, {C_PORT}, {C_DATABASE} or {C_DBNAME} in connection parameters")
        conn_params[C_HOST], conn_params[C_PORT], conn_params[C_DATABASE] = parse_pg_url(conn_params[C_URL])
        del conn_params[C_URL]
        if C_DRIVER in conn_params:
            del conn_params[C_DRIVER]

    # remove all records if we're overwriting the table
    if a_overwrite:
        print_verbose(1, a_verbose_level, f'truncating {a_dest_table}...')
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute(f"truncate table {a_dest_table};")
        conn.commit()
        cur.close()
        conn.close()

    # find string fields: they should be treated separately to speed up the process
    string_fields = set()
    for c in a_df.schema.fields:
        if isinstance(c.dataType, StringType):
            string_fields.add(c.name)

    # write each partition separately
    print_verbose(1, a_verbose_level, f'writing dataframe partitions to {a_dest_table}')
    a_df.foreachPartition(pg_make_copy_to_func(a_df.columns, string_fields, a_dest_table, **conn_params))
    print_verbose(1, a_verbose_level, 'done pg_copy.')


def get_fs_writer(a_df: DataFrame, a_format: str, a_header, a_overwrite, a_partition_by, a_verbose_level=3, a_compression=None, **a_options):
    """Constructs file system dataframe writer.

    :param a_df: the dataframe
    :param a_format: the format to save. Must be one of these: C_C_FS_SUPPORTED_FORMATS
    :param a_header: True/False - to write csv header or not.
    :param a_overwrite: Should we overwrite the data or append to it.
    :param a_partition_by: if set, it will do partitioned write
    :param a_verbose_level: printing verbosity level
    :param a_options: other options that will go to df.write.option(k, v)

    :return: dataframe writer
    """
    if a_format not in C_FS_SUPPORTED_FORMATS:
        raise ValueError(f"a_format must be one of the following: {C_FS_SUPPORTED_FORMATS}")
    if a_format in [C_CSV_GZ, C_CSV]:
        if not isinstance(a_header, bool):
            raise ValueError(f"For formats {C_CSV_GZ} and {C_CSV} you need to set a_header=True/False explicitly")
    else:
        # for non-csv format there should be no header or schema
        if isinstance(a_header, bool):
            raise ValueError(f"You should not set a_header for {a_format} format")

    # handle file format: parse csv.gz separately
    fmt = C_FORMAT_MAP.get(a_format, a_format)
    res = a_df.write.format(fmt)
    print_verbose(4, a_verbose_level, f"format: {fmt}")

    # setting the mode
    mode = "overwrite" if a_overwrite else "append"
    res = res.mode(mode)
    print_verbose(4, a_verbose_level, f"mode={mode}")

    if a_partition_by:
        if isinstance(a_partition_by, list):
            res = res.partitionBy(*a_partition_by)
        else:
            res = res.partitionBy(a_partition_by)

    if a_compression is not None:
        res = res.option("compression", a_compression)

    if a_format in {C_CSV_GZ, C_JSON_GZ}:
        print_verbose(4, a_verbose_level, f"option codec={C_GZIP_CODEC}")
        res = res.option("codec", C_GZIP_CODEC)

    if a_format in [C_CSV_GZ, C_CSV]:
        # https://issues.apache.org/jira/browse/SPARK-18579
        print_verbose(4, a_verbose_level, "options ignoreTrailingWhiteSpace=False, ignoreTrailingWhiteSpace=False")
        res = res.option("ignoreLeadingWhiteSpace", False).option("ignoreTrailingWhiteSpace", False)

    if a_header:
        print_verbose(4, a_verbose_level, "option header=True")
        res = res.option("header", True)

    # other options
    if a_options:
        print_verbose(4, a_verbose_level, f"other options: {hide_sensitive_str(a_options)}")
        for k, v in a_options.items():
            res = res.option(k, v)

    return res


def get_jdbc_writer(a_df: DataFrame, a_url: str, a_table: str, a_overwrite: bool, a_verbose_level=3, **a_options):
    """Constructs the dataframe writer to JDBC.

    :param a_df: the dataframe
    :param a_url: JDBC connection string
    :param a_table: table name
    :param a_overwrite: if True, the table will be overwritten
    :param a_verbose_level: print verbosity level
    :param a_options: other options that will go to df.write.option(k, v)
    :return:
    """
    res = a_df.write.format("jdbc")

    print_verbose(4, a_verbose_level, f"url: {a_url}")
    res = res.option('url', a_url)

    print_verbose(4, a_verbose_level, f"dbtable: {a_table}")
    res = res.option("dbtable", a_table)

    mode = "overwrite" if a_overwrite else "append"
    print_verbose(4, a_verbose_level, f"mode={mode}")
    res = res.mode(mode)

    if a_options:
        print_verbose(4, a_verbose_level, f"other options: {hide_sensitive_str(a_options)}")
        for k, v in a_options.items():
            res = res.option(k, v)

    return res


def save(a_df: DataFrame, a_dest_conn_info: dict, a_table,
         a_format=None, a_header=None,
         a_overwrite=True, a_fast_write=True,
         a_partition_by=None,
         a_columns_lowercase=False,
         a_columns_rename_map: dict = None,
         a_verbose_level=3):
    """ Saves a dataframe.

    :param a_df: the dataframe to be saved
    :param a_dest_conn_info: connection information for the destination
    :param a_table: table name in the database.
            Important! for DFS or local filesystems that will be a FOLDER always, so even if you specify
            a filename like file1.csv, it will make a folder named "file1.csv"
    :param a_format: must be set to file systems only and can be one of these: C_C_FS_SUPPORTED_FORMATS
    :param a_header: True/False - to write csv header or not
    :param a_overwrite: if True, the destination will be overwritten, otherwise append will happen
    :param a_fast_write:  if this is True, it will try to use native bulk copy routines;
            at this moment it is implemented only for Postgres, but it can be implemented for other SQL vendors
    :param a_partition_by: can be a list of str. Used to artition the dataframe by this list of columns or a single column
    :param a_columns_lowercase:  set it to True in the case if you explicitly want to set all columns to lower case before saving
            useful for PostgreSQL copy mode (a_fast_write=True)
    :param a_columns_rename_map: set some custom mapping of column names to convert them before saving
    :param a_verbose_level: print verbosity level
    :return: nothing
    """
    # in the case if the table name is a dict - get the format, header and schema from there, and check
    # if they are NOT set in the regular parameters
    a_compression = None
    if isinstance(a_table, dict):
        if a_format is not None or a_header is not None:
            raise ValueError("If you specify the a_table to be a dictionary of values, then you should NOT specify "
                             "a_format, a_header or a_partition_by separately because they should be specified in the table.")
        a_format = a_table.get("format", None)
        a_header = a_table.get("header", None)
        a_partition_by = a_table.get("partition_by", None)
        a_compression = a_table.get("compression", None)
        a_table = a_table.get("name", None)  # must be the last line
    elif not isinstance(a_table, str):
        raise ValueError("a_table must be either dict(name, format, header) or string")

    if a_table is None:
        raise ValueError("The a_table must be set (or a_table.name must be set, if a_table is a dictionary)")

    # parameters compatibility checks
    if a_columns_lowercase and a_columns_rename_map:
        raise ValueError("You cannot set a_columns_lowercase=True and a_columns_rename simultaneously")

    df = a_df  # further we will modify it - avoid using a_df in the code
    dest_type, dest_url, dest_options = parse_conn_info(a_dest_conn_info)

    # rename columns: this may be useful NOT only for Postgres, but also for file systems
    if a_columns_lowercase:
        df = lowercase_columns(df, a_verbose_level=a_verbose_level)
    if a_columns_rename_map:
        df = rename_columns(df, a_columns_rename_map, a_verbose_level=a_verbose_level)

    dtn = datetime.now()
    if dest_type == C_DFS or dest_type == C_LOCAL:
        path = join_path_generic(dest_url, a_table) if dest_type == C_DFS else join_path_os(dest_url, a_table)
        print_verbose(1, a_verbose_level, f"saving to {dest_type} path {path}, started at: {date_to_ymd_hms(dtn)}")
        get_fs_writer(df, a_format, a_header, a_overwrite, a_partition_by, a_verbose_level=a_verbose_level,
                      a_compression=a_compression, **dest_options).save(path)
    elif dest_type == C_JDBC:
        # for PostgreSQL we will use its fast copy function
        if a_fast_write and dest_url.lower().startswith("jdbc:postgresql"):
            print_verbose(1, a_verbose_level, f"saving dataframe to table {a_table} of {dest_url} (using PostgreSQL copy routine), started at: {date_to_ymd_hms(dtn)}")
            # forcibly set a_columns_lowercase=False, a_columns_rename=None because we already did this renaming before
            pg_copy(df, a_table, a_overwrite=a_overwrite, a_verbose_level=a_verbose_level,
                    # passing url separately: it is cut off from the options. But it will go to the **kwargs of the pg_copy
                    url=dest_url, **dest_options)
        else:
            print_verbose(1, a_verbose_level, f"saving to table {a_table} of {dest_url} (using regular JDBC), started at: {date_to_ymd_hms(dtn)}")
            get_jdbc_writer(df, dest_url, a_table, a_overwrite, a_verbose_level=a_verbose_level, **dest_options).save()
    else:
        raise NotImplemented(f"Type {dest_type} is not implemented yet")

    sp, dtn = seconds_passed(dtn, True)
    print_verbose(1, a_verbose_level, f"dataframe saved, seconds passed: {sp}, finished at: {date_to_ymd_hms(dtn)}")

# endregion


# region Other Dataframe routines

_CACHED_DF_LIST = []  # keep the list of cached dataframes


def ps(a_df: DataFrame):
    a_df.printSchema()


def cache(a_df: DataFrame, a_row_count=False, a_storage_level=StorageLevel.MEMORY_ONLY, a_temp_table=None, a_verbose_level=3):
    """Persists a dataframe.

    :param a_df: the dataframe
    :param a_row_count: perform counting of rows and force the dataframe to go into cache
    :param a_storage_level: the storage level. By default it will use memory only serialization.
            We stronly recommend using Kryo serialization enabled in Spark.
    :param a_verbose_level: print verbosity level

    :return: nothing
    """
    a_df.persist(storageLevel=a_storage_level)
    _CACHED_DF_LIST.append(a_df)
    cnt = -1
    if a_row_count:
        print_verbose(1, a_verbose_level, f"caching dataframe and counting rows")
        cnt = a_df.count()
        print_verbose(1, a_verbose_level, f"done caching, row count: {cnt:,}.")
    else:
        print_verbose(1, a_verbose_level, f"done caching.")
    if a_temp_table:
        a_df.createOrReplaceTempView(a_temp_table)
    return cnt


def unpersist(a_df: DataFrame, a_force=False, a_verbose_level=3):
    version_greater_240 = _spark.version > '2.4.0'
    if a_force or version_greater_240:
        a_df.unpersist()
        if a_df in _CACHED_DF_LIST:
            _CACHED_DF_LIST.remove(a_df)
    if a_force:
        print_verbose(1, a_verbose_level, f"dataframe is unpersisted forcibly")
    elif version_greater_240:
        print_verbose(1, a_verbose_level, f"dataframe is unpersisted")
    else:
        print_verbose(1, a_verbose_level, f"dataframe is NOT unpersisted because Spark version is less than 2.4.0, and it will lead to cascaded cache invalidation")


def uncache_all():
    """Uncaches previously serialized dataframes"""
    for df in _CACHED_DF_LIST:
        df.unpersist()
    del _CACHED_DF_LIST[:]


TEMP_TABLE_COUNTER = 0
TEMP_TABLES = []


def temp_table(a_df: DataFrame, a_name=None, a_prefix="t"):
    if a_name:
        a_df.createOrReplaceTempView(a_name)
        return a_name
    global TEMP_TABLE_COUNTER
    result = f"{a_prefix}_{TEMP_TABLE_COUNTER}"
    a_df.createOrReplaceTempView(result)
    TEMP_TABLES.append(result)
    TEMP_TABLE_COUNTER += 1
    return result


def sql(a_query, a_dfs=None, a_cache=False, a_row_count=False, a_temp_table=None, a_verbose_level=3) -> DataFrame:
    """

    :rtype: object
    """
    dfs = [] if a_dfs is None else a_dfs  # this is done to prevent mutable argument
    query = a_query
    # if a_dfs is passed, it means there must be {0}, []
    if dfs:
        dfs: list = dfs if isinstance(dfs, list) else [dfs]
        for i in range(len(dfs)):
            table_name = temp_table(dfs[i])
            query = query.replace("{" + str(i) + "}", table_name)
            query = query.replace("[" + str(i) + "]", table_name)
    print_verbose(3, a_verbose_level, query)
    result: DataFrame = _spark.sql(query)
    if a_cache and a_row_count:
        cache(result, a_row_count=True)
    elif a_row_count:
        count(result)
    elif a_cache:
        cache(result)
    if a_temp_table:
        result.createOrReplaceTempView(a_temp_table)
    return result


def distinct(a_df: DataFrame, a_columns, a_order=False, a_cache=False, a_row_count=False,
             a_return_count=False, a_verbose_level=3) -> Union[DataFrame, Tuple[DataFrame, int]]:
    if isinstance(a_columns, str):
        a_columns = [a_columns]
    table_name = temp_table(a_df)
    cols = comma_columns(a_columns)
    order_query = f" order by {cols}" if a_order else ""
    query = f"select distinct {cols} from {table_name}" + order_query
    df_result = sql(query, a_verbose_level=a_verbose_level)

    cnt = -1
    if a_cache:
        cnt = cache(df_result, a_row_count=a_row_count or a_return_count, a_verbose_level=a_verbose_level)

    if a_row_count or a_return_count:
        if cnt < 0:
            cnt = df_result.count()
            print_verbose(1, a_verbose_level, f"distinct row count (no caching): {cnt:,}")

    if a_return_count:
        return df_result, cnt
    return df_result


def distinct_values(a_df: DataFrame, a_column, a_order=False, a_verbose_level=3):
    rows = distinct(a_df, a_column, a_order, a_verbose_level=a_verbose_level).collect()
    return [r[a_column] for r in rows]


def min_value(a_df: DataFrame, a_column: str):
    pdf1 = sql(f"select min({a_column}) as {a_column} from [0]", a_df, a_verbose_level=5).toPandas()
    return pdf1[a_column].iloc[0]


def max_value(a_df: DataFrame, a_column: str):
    pdf1 = sql(f"select max({a_column}) as {a_column} from [0]", a_df, a_verbose_level=5).toPandas()
    return pdf1[a_column].iloc[0]


def collect_values(a_df: DataFrame, a_column, a_order=False, a_verbose_level=3):
    rows = change(a_df, a_select=[a_column], a_order_by=a_column if a_order else None, a_verbose_level=a_verbose_level).collect()
    return [r[a_column] for r in rows]


def count(a_df: DataFrame, a_cache=False, a_verbose_level=3):
    if not isinstance(a_cache, bool):
        raise ValueError("a_cache parameter must be bool")
    if a_cache:
        cache(a_df)
    cnt = a_df.count()
    print_verbose(1, a_verbose_level, f"row count: {cnt:,}")
    return cnt


def groupby_count(a_df: DataFrame, a_columns: ListOrStr, a_distinct_column=None, a_order_by=None, a_rc_desc=False, a_verbose_level=3) -> DataFrame:
    if isinstance(a_columns, str):
        a_columns = [a_columns]
    table_name = temp_table(a_df)
    cols = comma_columns(a_columns)
    order_by = a_order_by
    order_query = ""

    if order_by:
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(order_by, list):
            order_by = {c: "asc" for c in order_by}

    cnt_expression = "count(*)" if a_distinct_column is None else f"count(distinct {a_distinct_column})"
    rc_col_name = "RC" if a_distinct_column is None else f"DC_{a_distinct_column}"

    if a_rc_desc:
        if order_by is None:
            order_by = {rc_col_name: "desc"}
        else:
            order_by[rc_col_name] = "desc"
    order_query = " order by " + ", ".join(f"{c} {v}" for c, v in order_by.items()) if order_by else ""

    query = f"select {cols}, {cnt_expression} as {rc_col_name} from {table_name} group by {cols}" + order_query
    return sql(query, a_verbose_level=a_verbose_level)


def groupby_sum(a_df: DataFrame, a_group_columns: ListOrStr, a_sum_columns: ListOrStr, a_order=False, a_verbose_level=3) -> DataFrame:
    if isinstance(a_group_columns, str):
        a_group_columns = [a_group_columns]
    if isinstance(a_sum_columns, str):
        a_sum_columns = [a_sum_columns]
    table_name = temp_table(a_df)
    group_cols = comma_columns(a_group_columns) if a_group_columns else None
    sum_cols = ", ".join(f"SUM({x}) as SUM_{x}" for x in a_sum_columns)
    order_query = f" order by {group_cols}" if a_order and group_cols else ""
    query = f"select "
    if group_cols:
        query += f" {group_cols}, "
    query += f"{sum_cols} from {table_name} "
    if group_cols:
        query += f" group by {group_cols}"
    query += order_query
    return sql(query, a_verbose_level=a_verbose_level)


def groupby(a_df: DataFrame, a_group_columns: ListOrStr, a_columns: dict, a_order_by_group=False, a_order_by=None, a_verbose_level=3) -> DataFrame:
    if isinstance(a_group_columns, str):
        a_group_columns = [a_group_columns]
    if not isinstance(a_columns, dict):
        raise ValueError("a_coumns must be a dict of 'column': 'sum', 'column2': 'max', etc.")
    if a_order_by is not None and (not isinstance(a_order_by, dict) and not isinstance(a_order_by, list)) and not isinstance(a_order_by, str):
        raise ValueError("a_order_by must be a dictionary, list or string")
    table_name = temp_table(a_df)
    group_cols = comma_columns(a_group_columns)
    agg_cols = []
    for c, oplist in a_columns.items():
        if "," not in oplist:
            oplist = [oplist]
        else:
            oplist = oplist.split(",")
        for op in oplist:
            expr, colname = f"{op}({c})", f"{op}_{c}"
            if op == "RC":
                expr, colname = "count(*)", "RC"
            elif op == "DC":
                expr, colname = f"count(distinct {c})", f"DC_{c.replace(',', '_')}"
            agg_cols.append(f"{expr} as {colname}")
    agg_cols = ", ".join(agg_cols)
    order_query = ""
    if a_order_by_group:
        order_query = f" order by {group_cols}"
    elif a_order_by:
        if isinstance(a_order_by, dict):
            order_query = " order by " + ", ".join(f"{c} {o}" for c, o in a_order_by.items())
        else:
            if isinstance(a_order_by, str):
                a_order_by = [a_order_by]
            order_query = " order by " + ", ".join(f"{c} asc" for c in a_order_by)
    query = f"select {group_cols}, {agg_cols} from {table_name} group by {group_cols}" + order_query
    return sql(query, a_verbose_level=a_verbose_level)


def show(a_df: DataFrame, a_limit=100, a_t=False, a_row_count=False, a_verbose_level=3,
         a_where=None, a_order_by=None):
    df = a_df
    if a_row_count:
        cnt = df.count()
        print_verbose(1, a_verbose_level, f"total row count: {cnt:,}")
    if a_order_by or a_where:
        df = change(df, a_where=a_where, a_order_by=a_order_by, a_verbose_level=0)
    if a_limit > 0:
        print_verbose(1, a_verbose_level, f"showing top {a_limit} rows")
        df = df.limit(a_limit)
    pdf = df.toPandas()
    if a_t:
        pdf = pdf.T
    display_pdf(pdf)

BAD_SYMBOLS = '% ,()&-$#.'
HIVE_KEYWORDS_SET = {"ALL", "ALTER", "AND", "ARRAY", "AS", "AUTHORIZATION", "BETWEEN", "BIGINT", "BINARY", "BOOLEAN", "BOTH",
                     "BY", "CASE", "CAST", "CHAR", "COLUMN", "CONF", "CREATE", "CROSS", "CUBE", "CURRENT", "CURRENT_DATE", "CURRENT_TIMESTAMP",
                     "CURSOR", "DATABASE", "DATE", "DECIMAL", "DELETE", "DESCRIBE", "DISTINCT", "DOUBLE", "DROP", "ELSE", "END", "EXCHANGE",
                     "EXISTS", "EXTENDED", "EXTERNAL", "FALSE", "FETCH", "FLOAT", "FOLLOWING", "FOR", "FROM", "FULL", "FUNCTION", "GRANT",
                     "GROUP", "GROUPING", "HAVING", "IF", "IMPORT", "IN", "INNER", "INSERT", "INT", "INTERSECT", "INTERVAL", "INTO", "IS",
                     "JOIN", "LATERAL", "LEFT", "LESS", "LIKE", "LOCAL", "MACRO", "MAP", "MORE", "NONE", "NOT", "NULL", "OF", "ON", "OR",
                     "ORDER", "OUT", "OUTER", "OVER", "PARTIALSCAN", "PARTITION", "PERCENT", "PRECEDING", "PRESERVE", "PROCEDURE", "RANGE",
                     "READS", "REDUCE", "REVOKE", "RIGHT", "ROLLUP", "ROW", "ROWS", "SELECT", "SET", "SMALLINT", "TABLE", "TABLESAMPLE",
                     "THEN", "TIMESTAMP", "TO", "TRANSFORM", "TRIGGER", "TRUE", "TRUNCATE", "UNBOUNDED", "UNION", "UNIQUEJOIN", "UPDATE",
                     "USER", "USING", "UTC_TMESTAMP", "VALUES", "VARCHAR", "WHEN", "WHERE", "WINDOW", "WITH", "COMMIT", "ONLY", "REGEXP",
                     "RLIKE", "ROLLBACK", "START", "CACHE", "CONSTRAINT", "FOREIGN", "PRIMARY", "REFERENCES", "DAYOFWEEK", "EXTRACT",
                     "FLOOR", "INTEGER", "PRECISION", "VIEWS"}


def safe_col_name_simple(a_column):
    if a_column.startswith('`') and a_column.endswith('`'):
        return a_column
    if a_column.upper() in HIVE_KEYWORDS_SET:
        return f"`{a_column}`"
    for c in BAD_SYMBOLS:
        if c in a_column:
            return f"`{a_column}`"
    return a_column


def safe_col_name(a_col_name):
    if "." in a_col_name:
        parts = a_col_name.split(".")
        return ".".join(safe_col_name_simple(part) for part in parts)
    else:
        return safe_col_name_simple(a_col_name)


def comma_columns(a_fields, a_prefix="", a_separator=", "):
    if a_prefix == "":
        return a_separator.join(safe_col_name(c) for c in a_fields)
    return a_separator.join(f"{a_prefix}.{safe_col_name(c)}" for c in a_fields)


def get_change_query(a_input_table, a_input_cols,
                     a_drop=None, a_replace=None, a_rename=None, a_add=None, a_distinct=False,
                     a_filter_table=None, a_filter_columns=None, a_filter_not_table=None, a_filter_not_columns=None,
                     a_where=None, a_select_end=None, a_drop_end=None, a_order_by=None):
    """Generates a query for quick changin of the dataframe

    :param a_input_table: name of the main table
    :param a_input_cols: list of columns - passed as converted to safe
    :param a_drop:  list of columns to drop, e.g. ["column1", "column2"]
    :param a_replace:  must contain replacement expressions, e.e.  {"a": "log(a)"} - will give us "select log(a) as a"
    :param a_rename: {"a": "renamed_a", "b": "renamed_b"}}, so that it will make "select a as renamed_a"
    :param a_add:  must be a dict in format  {"x": "somevalue"}  so that it will add a new field "select somevalue as x"
    :param a_distinct: True or False (add "distinct" columns or not)

    :param a_filter_table: table name to use as filter
    :param a_filter_columns: the column to join for a_filter_table
    :param a_filter_not_table:  a table to check if the record is NOT in this table
    :param a_filter_not_columns: the column to join for a_filter_not_table

    :param a_where: where condition, e.g. "col1 = 20 and col2 < 10"
    :param a_select_end: columns to select after all filtering, where conditions are applied
    :param a_drop_end: columns to drop after all filtering, where conditions are applied
    :param a_order_by: order by clause in format {"column1": "asc", "column2": "desc"}
    """
    # to avoid problems with mutable arguments, we need this re-assignment
    if a_drop is None:
        a_drop = []
    if a_replace is None:
        a_replace = {}
    if a_rename is None:
        a_rename = {}
    if a_add is None:
        a_add = {}
    if a_drop_end is None:
        a_drop_end = {}
    if a_order_by is None:
        a_order_by = {}

    if a_filter_table and (not isinstance(a_filter_columns, list) or len(a_filter_columns) == 0):
        raise ValueError("when a_filter_table is not None, a_filter_columns should be a list")
    if a_filter_not_table and (not isinstance(a_filter_not_columns, list) or len(a_filter_not_columns) == 0):
        raise ValueError("when a_filter_not_table is not None, a_filter_not_columns should be some list")

        # initial list (not a dict - to preserve order)
    cols = [(c, c) for c in a_input_cols]
    # drop columns first
    a_drop = [safe_col_name(c) for c in a_drop]
    cols = [(c, v) for c, v in cols if c not in a_drop]
    # replace columns with expressions
    a_replace = {safe_col_name(k):v for k, v in a_replace.items()}
    cols = [(c, v if c not in a_replace else a_replace[c]) for c, v in cols]
    # rename columns
    a_rename = {safe_col_name(k):v for k, v in a_rename.items()}
    cols = [(c if c not in a_rename else a_rename[c], v) for c, v in cols]
    # add new columns
    cols.extend([(k, v) for k, v in a_add.items()])

    query = "select {}{} from {}".format(
        "distinct " if a_distinct else "",
        ", ".join(safe_col_name(c) if c == v else "{} as {}".format(v, safe_col_name(c)) for (c, v) in cols),
        a_input_table)

    if a_filter_table:
        query = "select x.* from (\n\t" \
                + query \
                + "\n) as x \ninner join " \
                + a_filter_table \
                + " f on " \
                + join_clause(a_filter_columns, "x", "f")

    where_is_added = False
    if a_filter_not_table:
        query = "select y.* from (\n\t" \
                + query \
                + "\n) as y \nleft join " \
                + a_filter_not_table \
                + " fn on " \
                + join_clause(a_filter_not_columns, "y", "fn") \
                + "\nwhere fn.{} is null".format(safe_col_name(a_filter_not_columns[0]))
        where_is_added = True

    if a_where:
        if where_is_added:
            query = "select * from (\n\t" + query + "\n) as z where {}".format(a_where)
        else:
            query = query + "\nwhere {}".format(a_where)

    if a_select_end and len(a_select_end) > 0:
        query = f"select {comma_columns(a_select_end)} from (\n\t" \
                + query \
                + "\n) as zs "

    if a_drop_end and len(a_drop_end) > 0:
        end_cols = [c for (c, v) in cols if c not in a_drop_end]
        query = f"select {comma_columns(end_cols)} from (\n\t" \
                + query \
                + "\n) as zd "

    if a_order_by and len(a_order_by) > 0:
        query = query + "\norder by {}".format(",".join("{} {}".format(safe_col_name(c), v) for c, v in a_order_by.items()))

    query = query + "\n"
    return query


def change(a_df: DataFrame, a_select=None, a_drop=None, a_replace=None, a_rename=None, a_add=None,
           a_distinct=False,
           a_filter_df=None, a_filter_columns=None, a_filter_not_df=None, a_filter_not_columns=None,
           a_where=None, a_select_end=None, a_drop_end=None, a_order_by=None,
           a_cache=False, a_row_count=False, a_temp_table=None,
           a_verbose_level=3, a_print_limit=None, a_ps=False):
    """Changes the existing dataframe by selecting subset of columns, dropping columns,
    replacing columns, renaming columns, adding columns, selecting distinct, filtering by another dataframe,
    adding a where-condition, etc.

    The arguments of this function are the same as of change_df_query()"""
    if a_drop is None:
        a_drop = []
    if a_replace is None:
        a_replace = {}
    else:
        # check correctness of replace keys:  all column names must be in the a_df.columns
        bad_column_names = [k for k in a_replace.keys() if k not in a_df.columns]
        if bad_column_names:
            raise ValueError(f"a_replace parameter contains columns which are missing in the dataframe: {bad_column_names}")
    if a_rename is None:
        a_rename = {}
    else:
        bad_column_names = [k for k in a_rename.keys() if k not in a_df.columns]
        if bad_column_names:
            raise ValueError(f"a_rename parameter contains columns which are missing in the dataframe: {bad_column_names}")
    if a_add is None:
        a_add = {}
    if a_drop_end is None:
        a_drop_end = {}
    if a_order_by is None:
        a_order_by = {}
    elif isinstance(a_order_by, str):
        a_order_by = {a_order_by: "asc"}
    elif isinstance(a_order_by, list):
        a_order_by = {c: "asc" for c in a_order_by}

    # printing of what operations are performed
    operations = []
    if a_select:
        operations.append("select")
    if a_drop:
        operations.append("drop")
    if a_replace:
        operations.append("replace")
    if a_rename:
        operations.append("rename")
    if a_add:
        operations.append("add")
    if a_distinct:
        operations.append("distinct")
    if a_filter_df:
        operations.append("filter_by_df")
    if a_filter_not_df:
        operations.append("filter_by_not_df")
    if a_where:
        operations.append("where")
    if a_select_end:
        operations.append("select_end")
    if a_drop_end:
        operations.append("drop_end")
    if a_order_by:
        operations.append("order_by")
    print_verbose(1, a_verbose_level, "changing dataframe, operations: " + ", ".join(operations))

    # check if a_select and a_select_end are all in dataframe, if not - raise exception
    # TODO:  think on what to do with columns that are structures with properties, e.g. event_properties.sceneId
    # if a_select:
    #     missing = missing_columns(a_select, a_df)
    #     if missing:
    #         raise ValueError(f"Bad a_select value: columns {missing} are missing in the dataframe. Full list of columns: {a_df.columns}")

    input_table = temp_table(a_df, a_prefix="df")
    filter_table = temp_table(a_filter_df, a_prefix="df_filter") if a_filter_df else None
    filter_not_table = temp_table(a_filter_not_df, a_prefix="df_not_filter") if a_filter_not_df else None

    input_cols = a_select if a_select else a_df.columns
    input_cols = [safe_col_name(c) for c in input_cols]

    query = get_change_query(input_table, input_cols,
                             a_drop=a_drop, a_replace=a_replace, a_rename=a_rename, a_add=a_add, a_distinct=a_distinct,
                             a_filter_table=filter_table, a_filter_columns=a_filter_columns, a_filter_not_table=filter_not_table, a_filter_not_columns=a_filter_not_columns,
                             a_where=a_where, a_select_end=a_select_end, a_drop_end=a_drop_end, a_order_by=a_order_by)

    if a_print_limit is not None and a_print_limit > 0:
        print_verbose(4, a_verbose_level, f"final query (firtst {a_print_limit} characters):\n" + query[:a_print_limit] + "... [cut off]")
    else:
        print_verbose(4, a_verbose_level, f"final query:\n" + query)
    result = sql(query, a_cache=a_cache, a_row_count=a_row_count, a_temp_table=a_temp_table, a_verbose_level=0)

    # we must do this post-processing (instead of pre-processing) because the new columns can be added at the end
    if a_select_end:
        missing = missing_columns(a_select_end, result)
        if missing:
            raise ValueError(f"Bad a_select_end value: columns {missing} are missing in the resulting dataframe. Full list of columns: {result.columns}")

    if a_ps:
        ps(result)

    return result


def join_clause(a_fields, a_left_prefix, a_right_prefix):
    return " and ".join("{}.{}={}.{}".format(a_left_prefix, safe_col_name(c), a_right_prefix, safe_col_name(c)) for c in a_fields)


def join(a_df1, a_df2, a_how="inner", a_join_cols=None, a_drop_cols1=None, a_drop_cols2=None,
         a_where=None, a_cache=False, a_unpersist1=False, a_unpersist2=False, a_row_count=False,
         a_old_count=None, a_return_count=False, a_temp_table=None, a_verbose_level=3):
    """Simplifies join between two dataframes.

    :param a_df1:  the first dataframe
    :param a_df2: the second dataframe
    :param a_how:  can be one of valid SQL join types:  "left", "right", "inner", "full outer"
    :param a_join_cols: columns on which we join. They must have the same names in both dataframes
    :param a_drop_cols1: columns to drop in the first dataframe after the join
    :param a_drop_cols2: columns to drop in the seconda dataframe after the join
    :param a_where: where condition to apply after the join
    :param a_cache: if True, the resulting dataframe will be cached
    :param a_unpersist1: if True, the first dataframe will be unpersisted
    :param a_unpersist2: if True, the second
    :param a_row_count: if True, the records will be counted. If also a_cache=True, it will immediately be in the cache.
    :param a_old_count: count of records of previous a_df1, if known.
    :param a_return_count: if True, the new record count will be returned witht he dataframe
    :param a_verbose_level: print verbosity level
    :return: a new dataframe or pair (dataframe, new_count) if a_return_count==True
    """
    if a_join_cols is None:
        raise ValueError("a_join_cols must be either string or a list of strings")
    alllowed_joins = ["left", "right", "inner", "full outer"]
    a_how = a_how.lower()
    if a_how not in alllowed_joins:
        raise ValueError(f"The join type {a_how} is not in allowed joins: {alllowed_joins}")
    if isinstance(a_join_cols, str):
        a_join_cols = [a_join_cols]
    if isinstance(a_drop_cols1, str):
        a_drop_cols1 = [a_drop_cols1]
    if isinstance(a_drop_cols2, str):
        a_drop_cols2 = [a_drop_cols2]

    table_name1 = temp_table(a_df1, a_prefix='df1')
    table_name2 = temp_table(a_df2, a_prefix='df2')

    # we could do this using sets, but this will lose column order
    # therefore below you may see this dummy way using lists
    cols1 = a_df1.columns if not a_drop_cols1 else [c for c in a_df1.columns if c not in a_drop_cols1]
    cols2 = a_df2.columns if not a_drop_cols2 else [c for c in a_df2.columns if c not in a_drop_cols2]

    join_cols = a_join_cols if a_join_cols else list(set(cols1).intersection(cols2))
    s_except1 = f"- all except {a_drop_cols1}" if a_drop_cols1 else ""
    s_except2 = f"- all except df1 and except {a_drop_cols2}" if a_drop_cols2 else ""

    # depending on the join type, we need to take the correct intersecion of columns
    if a_how == "inner" or a_how == "left":
        join_cols_list = [f"df1.{c}" for c in join_cols]
    elif a_how == "right":
        join_cols_list = [f"df2.{c}" for c in join_cols]
    elif a_how == "full outer":
        join_cols_list = [f"coalesce(df1.{c}, df2.{c}) as {c}" for c in join_cols]
    else:
        raise ValueError(f"The join type {a_how} is not in allowed joins: {alllowed_joins}")

    cols2_minus_cols1 = [c for c in cols2 if c not in cols1]
    first_cols = ["df1.{}".format(safe_col_name(c)) for c in cols1 if c not in join_cols]
    second_cols = ["df2.{}".format(safe_col_name(c)) for c in cols2_minus_cols1 if c not in join_cols]

    query = """
        select 
            -- join columns: 
            {}{}
        
            -- df1 columns{}:
            {}{} 

            -- df2 columns{}:
            {}
        from 
            {} df1
            {} join {} df2
                on {}
    """.format(", ".join(join_cols_list),
               "," if len(first_cols) > 0 or len(second_cols) > 0 else "",
               s_except1,
               ", ".join(first_cols),
               "," if len(second_cols) > 0 else "",
               s_except2,
               ", ".join(second_cols),
               table_name1,
               a_how,
               table_name2,
               join_clause(join_cols, 'df1', 'df2'))
    if a_where:
        query = "select * from (\n" + query + "\n) as x\n where \n" + a_where
    print_verbose(3, a_verbose_level, query)
    result = _spark.sql(query)

    new_cnt = -1
    if a_cache:
        print_verbose(1, a_verbose_level, "caching joined dataset")
        cache(result)
    if a_row_count or a_return_count:
        print_verbose(1, a_verbose_level, "counting records of the joined dataset...")
        new_cnt = result.count()
        if a_old_count:
            dropped = a_old_count - new_cnt
            dropped_percent = "({:.2%})".format(dropped / a_old_count)
            print_verbose(1, a_verbose_level, f"old record count: {a_old_count:,}; new record count: {new_cnt:,}; dropped {dropped:,} {dropped_percent} records.")
        else:
            print_verbose(1, a_verbose_level, f"record count: {new_cnt:,}")
    if a_unpersist1:
        print_verbose(1, a_verbose_level, "unpersisting the 1st dataframe")
        unpersist(a_df1)
    if a_unpersist2:
        print_verbose(1, a_verbose_level, "unpersisting the 2nd dataframe")
        unpersist(a_df2)
    print_verbose(1, a_verbose_level, "done join.")

    if a_temp_table:
        result.createOrReplaceTempView(a_temp_table)

    if a_return_count:
        return result, new_cnt
    return result


def union(a_df_list: list, a_columns: list=None, a_verbose_level=3, a_type=""):
    schema_types = None

    # check if everything is all right with the schemas: the datatypes must be the same
    for i, df in enumerate(a_df_list):
        if schema_types is None:
            schema_types = [str(f.dataType) for f in df.schema.fields if a_columns is None or f.name in a_columns]
        else:
            schema_types2 = [str(f.dataType) for f in df.schema.fields if a_columns is None or f.name in a_columns]
            if schema_types2 != schema_types:
                raise ValueError(f"Schema of the {i+1}th dataframe is different. Expected: {schema_types}, actual: {schema_types2}")

    cols = "*" if not a_columns else ", ".join(a_columns)
    temp_tables = [temp_table(df) for df in a_df_list]

    query = f"\nunion {a_type}\n".join([f"select {cols} from {t}" for t in temp_tables])
    return sql(query, a_verbose_level=a_verbose_level)


def union_all(a_df_list: list, a_columns: list=None, a_verbose_level=3):
    return union(a_df_list, a_columns, a_verbose_level, a_type="all")


def moving_average(a_df, a_group_col, a_value_col, a_sort_col, a_moving_avg_col, a_window, a_min_periods=1, a_return_all_columns=True):
    # DO NOT REMOVE select(*a_df.columns):  this is related to inability to add column dynamically to it
    schema = a_df.select(*a_df.columns).schema if a_return_all_columns else a_df.select(a_group_col, a_sort_col, a_value_col).schema
    schema = (schema.add(StructField(a_moving_avg_col, DoubleType())))

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def ma(a_pdf):
        a_pdf[a_moving_avg_col] = a_pdf.sort_values(a_sort_col)[a_value_col].rolling(window=a_window, min_periods=a_min_periods).mean()
        return a_pdf

    return a_df.groupby(a_group_col).apply(ma)


def exp_moving_average(a_df, a_group_col, a_value_col, a_sort_col, a_moving_avg_col, a_span, a_min_periods=1, a_return_all_columns=True):
    # DO NOT REMOVE select(*a_df.columns):  this is related to inability to add column dynamically to it
    schema = a_df.select(*a_df.columns).schema if a_return_all_columns else a_df.select(a_group_col, a_sort_col, a_value_col).schema
    schema = (schema.add(StructField(a_moving_avg_col, DoubleType())))

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def ema(a_pdf):
        a_pdf[a_moving_avg_col] = a_pdf.sort_values(a_sort_col)[a_value_col].ewm(span=a_span, min_periods=a_min_periods).mean()
        return a_pdf

    return a_df.groupby(a_group_col).apply(ema)


def median(a_df, a_part_columns, a_column, a_new_column_name, a_row_count=False, a_cache=False):
    if isinstance(a_part_columns, str):
        a_part_columns = [a_part_columns]
    w = Window.partitionBy(*a_part_columns).orderBy(a_column)
    df_result = a_df.withColumn(
        "rank", sf.row_number().over(w)
    ).withColumn(
        "count_row_part",
        sf.count(a_column).over(Window.partitionBy(a_part_columns))
    ).withColumn(
        "even_flag",
        sf.when(
            sf.col("count_row_part") % 2 == 0,
            sf.lit(1)
        ).otherwise(
            sf.lit(0)
        )
    ).withColumn(
        "mid_value",
        sf.floor(sf.col("count_row_part") / 2)
    ).withColumn(
        "avg_flag",
        sf.when(
            (sf.col("even_flag") == 1) & (sf.col("rank") == sf.col("mid_value"))
            |
            ((sf.col("rank") - 1) == sf.col("mid_value")),
            sf.lit(1)
        ).otherwise(
            sf.when(
                sf.col("rank") == sf.col("mid_value") + 1,
                sf.lit(1)
            )
        )
    ).filter(
        sf.col("avg_flag") == 1
    ).drop(
        "avg_flag"
    ).groupby(
        a_part_columns
    ).agg(
        sf.avg(sf.col(a_column)).alias(a_new_column_name)
    )
    if a_cache and a_row_count:
        count(df_result, a_cache=True)
    elif a_cache:
        cache(df_result)
    elif a_row_count:
        count(df_result, a_cache=False)
    return df_result


def check_if_statistics_are_correct(a_stats):
    known_stats = {"mean",
                    "std",
                    "min",
                    "max",
                    "skew",
                    "kurtosis",
                    "mean_minus_1std",
                    "mean_minus_2std",
                    "mean_minus_3std",
                    "mean_plus_1std",
                    "mean_plus_2std",
                    "mean_plus_3std",
                    "q25_minus_15iqr",
                    "q75_plus_15iqr",
                    "sum",
                    "count"}
    unknown_stats = [s for s in a_stats if s not in known_stats]
    if unknown_stats:
        raise ValueError(f"Unknown a_stats: {unknown_stats}")


def calc_stat_local(a_pdf,
                    a_stats,
                    a_quantiles,
                    a_column_prefix_map,
                    a_group_columns,
                    a_return_uppercase=False,
                    a_reset_index=True,
                    a_round_to_decimal_places=-1,
                    a_trunc_to_decimal_places=-1
                    ):
    if a_round_to_decimal_places != -1 and a_trunc_to_decimal_places != -1:
        raise ValueError("yout should use either a_round_to_decimal_places or a_trunc_to_decimal_places, but not both")
    # region indicators of which statistics to calculate; they will go as closures to the stat_func()
    if isinstance(a_group_columns, str):
        a_group_columns = [a_group_columns]
    calc_count = "count" in a_stats
    calc_ms = "mean_minus_1std" in a_stats or "mean_minus_2std" in a_stats or "mean_minus_3std" in a_stats \
              or "mean_plus_1std" in a_stats or "mean_plus_2std" in a_stats or "mean_plus_3std" in a_stats
    calc_mean = "mean" in a_stats or calc_ms
    calc_std = "std" in a_stats or calc_ms
    calc_sum = "sum" in a_stats
    calc_min = "min" in a_stats
    calc_max = "max" in a_stats
    calc_skew = "skew" in a_stats
    calc_kurtosis = "kurtosis" in a_stats
    calc_mean_minus_1std = "mean_minus_1std" in a_stats
    calc_mean_minus_2std = "mean_minus_2std" in a_stats
    calc_mean_minus_3std = "mean_minus_3std" in a_stats
    calc_mean_plus_1std = "mean_plus_1std" in a_stats
    calc_mean_plus_2std = "mean_plus_2std" in a_stats
    calc_mean_plus_3std = "mean_plus_3std" in a_stats
    calc_iqr = "q25_minus_15iqr" in a_stats or "q75_plus_15iqr" in a_stats
    # in the case if IQR is requested, we need to appen 0.25 and 0.75 to the list of quantiles
    calc_quantiles = a_quantiles if not calc_iqr else sorted(list(set(a_quantiles).union({0.25, 0.75})))

    # endregion

    # region prepare list of pointers on statistics function
    def count_func():
        def f(x):
            return len(x)

        f.__name__ = "count"
        return f

    def min_func():
        def f(x):
            return np.min(x)

        f.__name__ = "min"
        return f

    def max_func():
        def f(x):
            return np.max(x)

        f.__name__ = "max"
        return f

    def quantile_func(a_q):
        def f(x):
            return np.quantile(x, a_q)

        f.__name__ = "q" + f"{a_q:.2f}"[2:]
        return f

    def std(a):
        return np.std(a)


    def skew(a):
        return scipy_skew(a)

    def kurtosis(a):
        return scipy_kurtosis(a)

    agg_funcs = []
    if calc_mean:
        agg_funcs.append(np.mean)
    if calc_std:
        agg_funcs.append(std)
    if calc_sum:
        agg_funcs.append(np.sum)
    if calc_count:
        agg_funcs.append(count_func())
    if calc_min:
        agg_funcs.append(min_func())
    if calc_max:
        agg_funcs.append(max_func())
    if calc_skew:
        agg_funcs.append(skew)
    if calc_kurtosis:
        agg_funcs.append(kurtosis)

    if calc_quantiles:
        for q in calc_quantiles:
            agg_funcs.append(quantile_func(q))
    # endregion

    # the final list of columns to return
    get_cols = a_stats + ["q" + f"{q:.2f}"[2:] for q in a_quantiles]

    pdf_res_all = None
    for c, p in a_column_prefix_map.items():
        pdf_res = a_pdf[a_group_columns + [c]]
        pdf_res: pd.DataFrame = pdf_res.groupby(a_group_columns).agg(agg_funcs)
        pdf_res.columns = pdf_res.columns.droplevel(0)

        # these columns can be pre-computed based on other statistics
        if calc_mean_minus_1std:
            pdf_res["mean_minus_1std"] = pdf_res["mean"] - pdf_res["std"]
        if calc_mean_plus_1std:
            pdf_res["mean_plus_1std"] = pdf_res["mean"] + pdf_res["std"]
        if calc_mean_minus_2std:
            pdf_res["mean_minus_2std"] = pdf_res["mean"] - 2 * pdf_res["std"]
        if calc_mean_plus_2std:
            pdf_res["mean_plus_2std"] = pdf_res["mean"] + 2 * pdf_res["std"]
        if calc_mean_minus_3std:
            pdf_res["mean_minus_3std"] = pdf_res["mean"] - 3 * pdf_res["std"]
        if calc_mean_plus_3std:
            pdf_res["mean_plus_3std"] = pdf_res["mean"] + 3 * pdf_res["std"]
        if calc_iqr:
            pdf_res["q25_minus_15iqr"] = pdf_res["q25"] - 1.5 * (pdf_res["q75"] - pdf_res["q25"])
            pdf_res["q75_plus_15iqr"] = pdf_res["q75"] + 1.5 * (pdf_res["q75"] - pdf_res["q25"])

        # keep just statistics + quantiles that we requested (but do not take those which served as basis for others)
        pdf_res = pdf_res[get_cols]

        if a_round_to_decimal_places >= 0:
            for c2 in get_cols:
                pdf_res[c2] = np.around(pdf_res[c2].values, a_round_to_decimal_places)
        if a_trunc_to_decimal_places >= 0:
            multiplier = 10 ** a_trunc_to_decimal_places
            for c2 in get_cols:
                pdf_res[c2] = np.trunc(pdf_res[c2].values * multiplier) / multiplier

        # rename columns - append the prefix
        pdf_res.columns = [p + c for c in pdf_res.columns]

        pdf_res_all = pdf_res if pdf_res_all is None else pd.merge(pdf_res_all, pdf_res, left_index=True, right_index=True)

    if a_return_uppercase:
        pdf_res_all.columns = [c.upper() for c in pdf_res_all.columns]

    if a_reset_index:
        return pdf_res_all.reset_index()  # groupping columns will remain the same case
    return pdf_res_all


def statistics(a_df,
                a_group_columns,
                a_column_prefix_map,
                a_stats=None,
                a_quantiles=None,
                a_return_schema=False,
                a_split_group_columns=None, a_return_uppercase=False, a_round_to_decimal_places=-1, a_trunc_to_decimal_places=-1):
    # region check parameters
    if a_stats is None and a_quantiles is None:
        raise ValueError("Please, specify either a_stats or a_quantiles")
    if isinstance(a_group_columns, str):
        a_group_columns = [a_group_columns]
    if isinstance(a_split_group_columns, str):
        a_split_group_columns = [a_split_group_columns]
    stats = [] if a_stats is None else [s.lower() for s in a_stats]
    quantiles = [] if a_quantiles is None else a_quantiles
    check_if_statistics_are_correct(stats)
    # endregion

    # region schema
    field_names = []
    for _, p in a_column_prefix_map.items():
        field_names.extend([p + f for f in stats] + [p + "q" + f"{q:.2f}"[2:] for q in quantiles])
    output_cols = get_schema_columns(a_df, a_group_columns)
    if a_return_uppercase:
        field_names = [f.upper() for f in field_names]
    output_cols.extend([StructField(f, DoubleType(), True) for f in field_names])
    # endregion

    def stat_func2(a_pdf):
        return calc_stat_local(a_pdf, stats, quantiles, a_column_prefix_map, a_group_columns,
                               a_return_uppercase=a_return_uppercase, a_reset_index=True,
                               a_round_to_decimal_places=a_round_to_decimal_places,
                               a_trunc_to_decimal_places=a_trunc_to_decimal_places)

    schema = StructType(output_cols)

    df_result = a_df.groupBy(a_split_group_columns if a_split_group_columns else a_group_columns).applyInPandas(
        stat_func2, schema)

    if a_return_schema:
        return df_result, schema
    return df_result


def get_schema_columns(a_df, a_columns=None):
    if a_columns is None:
        a_columns = a_df.columns
    elif isinstance(a_columns, str):
        a_columns = [a_columns]
    schema_dict = {c.name: c for c in a_df.schema}
    result = []
    for gc in a_columns:
        if gc not in schema_dict:
            raise ValueError(f"Column {gc} is not found in the schema")
        result.append(schema_dict[gc])
    return result

def get_schema_columns_string(a_df, a_columns=None):
    if a_columns is None:
        a_columns = a_df.columns
    elif isinstance(a_columns, str):
        a_columns = [a_columns]
    schema_dict = {c.name: c for c in a_df.schema}
    result = []
    for gc in a_columns:
        if gc not in schema_dict:
            raise ValueError(f"Column {gc} is not found in the schema")
        result.append(schema_dict[gc].name + " " + str(schema_dict[gc].dataType).lower()[:-4])
    return ", ".join(result)

def correlation(a_df: DataFrame, a_group_columns, a_column1, a_column2, a_corr_column, a_type="spearman"):
    known_corr_types = ["spearman", "pearson"]
    if a_type.lower() not in known_corr_types:
        raise ValueError(f"Unknown correlation type {a_type}, possible types: {known_corr_types}")

    # the schema is NOT a dictionary: it is a list; we need to create the columns in the same order as they are enumerated in a_group_columns
    output_cols = get_schema_columns(a_df, a_group_columns)
    output_cols.append(StructField(a_corr_column, DoubleType(), True))

    @pandas_udf(StructType(output_cols), PandasUDFType.GROUPED_MAP)
    def corr_func(a_pdf):
        # generate the key
        row = [a_pdf[gc].iloc[0] for gc in a_group_columns]

        # calculate the statistics
        if a_type == "spearman":
            corr = spearmanr(a_pdf[a_column1], a_pdf[a_column2])[0]
        elif a_type == "pearson":
            corr = pearsonr(a_pdf[a_column1], a_pdf[a_column2])[0]
        else:
            raise ValueError(f"Unknown correlation type {a_type}")

        row.append(corr)
        return pd.DataFrame([row], columns=a_group_columns + [a_corr_column])

    return a_df.groupBy(a_group_columns).apply(corr_func)


# endregion


# region Data Investigation routines

def get_dc(a_df):
    """Returns distinct counts for every column"""
    result_list = []
    t = temp_table(a_df)
    for c in a_df.columns:
        q_distinct = f"select count(distinct {c}) as {c} from {t}"
        print(f"Running query {q_distinct}")
        df_distinct = sql(q_distinct).toPandas()
        result_list.append((df_distinct.columns[0], df_distinct.iloc[0, 0], q_distinct))
    return pd.DataFrame(result_list, columns=["COL_NAME", "DC", "query"])


def show_values(a_df, a_pdf_dc=None, a_limit=10):
    """Shows the distinct values on all columns for a dataframe"""
    t = temp_table(a_df)
    if a_pdf_dc is None:
        pdf_dc = get_dc(a_df)
        pdf = pdf_dc[pdf_dc["DC"]>0]
        print("Distinct counts:")
        display_pdf(pdf)
    else:
        print("Distinct count table received:")
        pdf = a_pdf_dc
        display_pdf(pdf)

    for f, dc in zip(pdf["COL_NAME"].values, pdf["DC"].values):
        if dc > 0 and dc < 100000:
            pdf_g = sql(f"select {f}, count(*) from {t} group by {f} order by count(*) desc").toPandas()
            if dc > a_limit:
                print(f"Showing first {a_limit} values of {f} (total {dc} distinct values):")
                display_pdf(pdf_g.head(a_limit))
            else:
                print(f"Values of {f} (total {dc} distinct values):")
                display_pdf(pdf_g)
        else:
            print(f"Field {f} has dc={dc}, we won't be fetching info about it")

# endregion


# region Unit test routines

def ut_missing_key(a_stop, a_df_main, a_df_lookup, a_column_main, a_column_lookup=None, a_show=False, a_limit=100, a_verbose_level=3, a_return_df=False):
    if not a_column_lookup:
        a_column_lookup = a_column_main
    print_verbose(1, a_verbose_level, f"unit test: checking if values for {a_column_main} in main datafrare are missing in values {a_column_lookup} of the lookup")
    table_main = temp_table(a_df_main, a_prefix='t_main')
    table_lookup = temp_table(a_df_lookup, a_prefix='t_lookup')
    cnt = sql(f"""
        select count(*) 
        from 
            {table_main} m 
            left join {table_lookup} l on m.{a_column_main} = l.{a_column_lookup} 
        where 
            l.{a_column_lookup} is null
    """).collect()[0][0]
    if cnt > 0:
        err = f"there are {cnt:,} rows with bad value of {a_column_main}"
        print_verbose(1, a_verbose_level, err)
        if a_show:
            df_bad = sql(f"""
                select * 
                from 
                    {table_main} m 
                    left join {table_lookup} l on m.{a_column_main} = l.{a_column_lookup} 
                where 
                    l.{a_column_lookup} is null
            """)
            show(df_bad, a_limit=a_limit, a_verbose_level=a_verbose_level)
        if a_stop:
            raise Exception(C_TEST_FAILURE + err)
        if a_return_df:
            return True, df_bad
        return False
    else:
        print_verbose(1, a_verbose_level, C_TEST_PASSED)
        if a_return_df:
            return True, None
        return True


def ut_check_duplicates(a_stop, a_df, a_columns, a_show=False, a_limit=100, a_verbose_level=3, a_return_dup_df=False):
    if isinstance(a_columns, str):
        a_columns = [a_columns]
    print_verbose(1, a_verbose_level, "unit test: checking for duplicates")
    table_name = temp_table(a_df)
    cols = ', '.join(a_columns)
    df_grouped = sql(f"select {cols}, count(*) as RC from {table_name} group by {cols} having RC > 1", a_verbose_level=a_verbose_level)
    cnt = df_grouped.count()
    if cnt > 0:
        err = f"found {cnt:,} duplicates"
        str_add = ", showing {}:".format("all" if a_limit <= 0 else f"first {a_limit}") if a_show else ""
        print_verbose(1, a_verbose_level, err + str_add)
        if a_show:
            show(df_grouped, a_limit=a_limit, a_verbose_level=a_verbose_level)
        if a_stop:
            raise Exception(C_TEST_FAILURE + err)
        if a_return_dup_df:
            return False, df_grouped
        return False
    else:
        print_verbose(1, a_verbose_level, C_TEST_PASSED)
        if a_return_dup_df:
            return True, None
        return True


def ut_check_dependent_columns(a_stop, a_df, a_parent_columns, a_children_columns, a_show=False, a_limit=10,
                               a_verbose_level=3, a_return_bad_df=False,
                               a_show_hist=False):
    if isinstance(a_children_columns, str):
        a_children_columns = [a_children_columns]
    if isinstance(a_parent_columns, str):
        a_parent_columns = [a_parent_columns]
    parent_cols = ', '.join(a_parent_columns)
    children_cols = ', '.join(a_children_columns)
    print_verbose(1, a_verbose_level, f"unit test: checking for relationship [{parent_cols}] -> [{children_cols}]")

    table_name = temp_table(a_df)
    df_bad_rows = sql(f"select {parent_cols}, count(distinct {children_cols}) as RC from {table_name} group by {parent_cols} having RC > 1 order by RC desc", a_verbose_level=a_verbose_level)
    cnt = df_bad_rows.count()
    if cnt > 0:
        err = f"found {cnt:,} rows with non-working relationship [{parent_cols}] -> [{children_cols}]"
        str_add = ", showing {}:".format("all" if a_limit <= 0 else f"first {a_limit}") if a_show else ""
        print_verbose(1, a_verbose_level, err + str_add)
        if a_show:
            show(df_bad_rows, a_limit=a_limit, a_verbose_level=a_verbose_level)
        if a_show_hist:
            pdf_hist = groupby_count(change(df_bad_rows, a_rename={"RC": "row_count"}),
                                     "row_count").toPandas().sort_values("row_count")
            pdf_hist.plot.bar(x="row_count", y="RC")
        if a_stop:
            raise Exception(C_TEST_FAILURE + err)
        if a_return_bad_df:
            return False, df_bad_rows
        return False
    else:
        print_verbose(1, a_verbose_level, C_TEST_PASSED)
        if a_return_bad_df:
            return True, None
        return True


def ut_null_check(a_stop, a_df, a_columns, a_verbose_level=3):
    print_verbose(1, a_verbose_level, f"unit test: checking for nulls for columns {a_columns}")
    cnt = sql("select count(*) from {0} where " + " or ".join(f"{c} is null" for c in a_columns), a_df, a_verbose_level=a_verbose_level).collect()[0][0]
    if cnt > 0:
        err = f"found {cnt:,} rows with nulls in one of these columns"
        print_verbose(1, a_verbose_level, err)
        if a_stop:
            raise Exception(C_TEST_FAILURE + err)
        return False
    else:
        print_verbose(1, a_verbose_level, C_TEST_PASSED)
        return True


def ut_schemas_equal(a_stop, a_df1: DataFrame, a_df2: DataFrame, a_check_nulls=False, a_verbose_level=3):
    str1 = str(a_df1.schema)
    str2 = str(a_df2.schema)
    if not a_check_nulls:
        str1 = str1.replace("false", "true")
        str2 = str2.replace("false", "true")
    if str1 != str2:
        print_verbose(1, a_verbose_level, f"Schemas of the dataframes are different:\ndf1\n:{df_schema_to_str(a_df1)},\n\ndf2:\n{df_schema_to_str(a_df2)}")
        if a_stop:
            raise Exception("Dataframes have different schemas")
    else:
        print_verbose(1, a_verbose_level, C_TEST_PASSED)
        return True

# endregion


# region Row/Column/Schema manipulation routines

def row_to_str(a_row: Row):
    if hasattr(a_row, "__fields__"):
        return "Row(%s)" % ", ".join("%s=%r" % (k, str(v))
                                     for k, v in zip(a_row.__fields__, tuple(a_row)))
    else:
        return "<Row(%s)>" % ", ".join(str(v) for v in tuple(a_row))


def df_schema_to_str(a_df: DataFrame):
    """Gets the schema string out of the dataframe.
    We won't use printSchema() just becase we need to use it with print_verbose() method often.
    :param a_df: dataframe to get the schema from it
    """
    return a_df._jdf.schema().treeString()


def schema_to_str(a_schema):
    """For df.schema (that will return StructType(list of IntegerType(), StringType()...)
    we need to construct a schema string with the same format as printSchema()
    """
    return "root\n |-- " + "\n |-- ".join(f"{f.name}: {str(f.dataType)[:-4].lower()} (nullable = {str(f.nullable).lower()})" for f in a_schema.fields)


def schema_to_code(a_schema, indent=1):
    if isinstance(a_schema, StructType):
        spacer = "StructType([\n%s\n" + "    " * (indent - 1) + "])"
        return (spacer % ",\n".join("    " * indent + schema_to_code(field, indent + 1) for field in a_schema))
    elif isinstance(a_schema, StructField):
        return "StructField(\"%s\", %s, %s)" % (a_schema.name, schema_to_code(a_schema.dataType, indent), str(a_schema.nullable))
    else:
        return str(a_schema) + "()"


def rename_columns(a_df: DataFrame, a_columns_rename_map: dict, a_verbose_level=3):
    """Renames columns of a dataframe using the renaming map.

    :param a_df: the dataframe
    :param a_columns_rename_map:  dict of old_name->new_name
    :param a_verbose_level: print verbosity level

    :return:
    """
    if not a_columns_rename_map:
        return a_df
    table_name = temp_table(a_df)
    cols_str = ", ".join(f"{k} as {a_columns_rename_map[k]}" if k in a_columns_rename_map else f"{k}" for k in a_df.columns)
    result = _spark.sql(f"select {cols_str} from {table_name}")
    print_verbose(1, a_verbose_level, "Columns renamed, new schema:\n{df_schema_to_str(df)}")
    return result


def lowercase_columns(a_df: DataFrame, a_verbose_level=3):
    """Converts all the columns of a dataframe to lowercase. Useful for writing into Postgres,
    where all identifiers (column names) are lowercased by default.

    :param a_df: the dataframe
    :param a_verbose_level: print verbosity level
    :return: column rename map  c->c.lower()
    """
    return rename_columns(a_df, {k: k.lower() for k in a_df.columns}, a_verbose_level=a_verbose_level)


# sensitive password strings that may be encountered in options
_SENSITIVE_KEYS = {"password", "pass", "passwd", "secret"}


def scramble_value(a_key: str, a_value):
    # if the key is equal to one of these - it is sensitive
    if a_key in _SENSITIVE_KEYS:
        return "(hidden)"
    # if key' part contains any of these - it is secret
    for s in _SENSITIVE_KEYS:
        if s in a_key:
            return "(hidden)"
    return a_value


def hide_sensitive_str(d: dict):
    """
    The main purpose is to hide the passwords when outputting sensitive information.
    :param d: the dictionary to output
    :return: string with "(hidden)" instead of real passwords
    """
    return "{" + ", ".join("{}={}".format(str(k), scramble_value(k, v)) for k, v in d.items()) + "}"


def missing_columns(a_columns: list, a_df: DataFrame, a_lowercase=False):
    """Returns those columns which are in the dataframe

    :param a_columns: list of columns to check
    :param a_df: the dataframe to look into
    :param a_lowercase: if True, we will compare by lowercase (but we will return original column names)
    :return: those columns which are missing inside the dataframe
    """
    if not a_lowercase:
        return [c for c in a_columns if c not in a_df.columns]
    return [c for c in a_columns if c.lower() not in [col.lower() for col in a_df.columns]]


def numpy_to_spark_type(c):
    if c == np.int8:
        return ByteType()
    elif c == np.int16:
        return ShortType()
    elif c == np.int32:
        return IntegerType()
    elif c == np.int64:
        return LongType()
    elif c == np.float32:
        return FloatType()
    elif c == np.float64:
        return DoubleType()
    elif np.issubdtype(c, np.datetime64):
        return TimestampType()
    elif c == np.object:
        return StringType()
    else:
        raise Exception(f"Unknown type {c}")


def spark_schema_from_pdf(a_pdf):
    fields = []
    for c, t in zip(a_pdf.columns, a_pdf.dtypes):
        fields.append(StructField(c, numpy_to_spark_type(t), True))
    return StructType(fields)


def create_spark_df(a_pdf: pd.DataFrame):
    return _spark.createDataFrame(a_pdf, spark_schema_from_pdf(a_pdf))

# endregion
