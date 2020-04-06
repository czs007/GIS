"""
Copyright (C) 2019-2020 Zilliz. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pyspark.sql import SparkSession

from app.common import db, utils
from arctern_pyspark import register_funcs

class Spark(db.DB):
    def __init__(self, db_config):
        envs = db_config['spark'].get('envs', None)
        if envs:    # for spark on yarn
            self._setup_driver_envs(envs)

        import uuid
        self._db_id = str(uuid.uuid1()).replace('-', '')
        self._db_id = "1"
        self._db_name = db_config['db_name']
        self._db_type = 'spark'
        self._reset_tables()

        print("init spark begin")
        import socket
        localhost_ip = socket.gethostbyname(socket.gethostname())
        _t = SparkSession.builder \
            .appName(db_config['spark']['app_name']) \
            .master(db_config['spark']['master-addr']) \
            .config('spark.driver.host', localhost_ip) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")

        configs = db_config['spark'].get('configs', None)
        if configs:
            for key in configs:
                print("spark config: {} = {}".format(key, configs[key]))
                _t = _t.config(key, configs[key])

        self.session = _t.getOrCreate()

        print("init spark done")
        register_funcs(self.session)

    def _reset_tables(self):
        self._table_list = []
        self._tables_meta = []
        self._tables_name = []
        self._tables_index = {}
        self._tables_parents = []

    def table_list(self):
        return self._table_list

    def _setup_driver_envs(self, envs):
        import os

        keys = ('PYSPARK_PYTHON', 'PYSPARK_DRIVER_PYTHON', 'JAVA_HOME',
                'HADOOP_CONF_DIR', 'YARN_CONF_DIR', 'GDAL_DATA', 'PROJ_LIB'
                )

        for key in keys:
            value = envs.get(key, None)
            if value:
                os.environ[key] = value

    def _create_session(self):
        """
        clone new session
        """
        session = self.session.newSession()
        register_funcs(session)
        return session

    def run(self, sql):
        """
        submit sql to spark
        """
        #session = self._create_session()
        return self.session.sql(sql)

    def run_for_json(self, sql):
        """
        convert the result of run() to json
        """
        _df = self.run(sql)
        return _df.coalesce(1).toJSON().collect()

    def unload(self, table_metas):
        """
            根据 已有的关系，推导出所有要 unload的表
        """
        self.session.catalog.dropGlobalTempView("nyc_taxi")
        # self.session.catalog.listTables()

    def _update_table_metas(self, tables_meta, is_replace=False):
        if is_replace:
            self._reset_tables()
            self._tables_meta = tables_meta
            self._process_tables_meta()
            return

    def load(self, table_metas):
        for meta in table_metas:
            if 'path' in meta and 'schema' in meta and 'format' in meta:
                options = meta.get('options', None)

                schema = str()
                for column in meta.get('schema'):
                    for key, value in column.items():
                        schema += key + ' ' + value + ', '
                rindex = schema.rfind(',')
                schema = schema[:rindex]

                df = self.session.read.format(meta.get('format')) \
                    .schema(schema) \
                    .load(meta.get('path'), **options)
                df.createOrReplaceTempView(meta.get('name'))
            elif 'sql' in meta:
                df = self.run(meta.get('sql', None))
                df.createOrReplaceTempView(meta.get('name'))
                #sql_str = "CACHE TABLE global_temp.%s"%meta.get('name')
                #print("cache ..... %s"%sql_str)
                #self.session.sql(sql_str)
            print("current database :", self.session.catalog.currentDatabase())
            if meta.get('visibility') == 'True':
                self._table_list.append('global_temp.' + meta.get('name'))
        self._tables_meta = table_metas
        self._process_tables_meta()
        self.my_test()

    def _process_tables_meta(self):
        for i, table in enumerate(self._tables_meta):
            table_name = table['name']
            self._tables_name.append(table_name)
            self._tables_index[table_name] = i

        for table in self._tables_meta:
            table_name = table['name']
            parents = table.get("parents", [])
            parent_indexs = set()
            for parent in parents:
                parent_indexs.add(self._tables_index[parent])
            self._tables_parents.append(parent_indexs)

        changed = list(range(0, len(self._tables_parents)))
        while (changed):
            new_changed = []
            for change in changed:
                parents = self._tables_parents[change]
                if parents:
                    new_parents = set()
                    for parent in parents:
                        new_parents |= self._tables_parents[parent]
                    if not new_parents.issubset(parents):
                        new_changed.append(change)
                        self._tables_parents[change] = new_parents | parents
            changed = new_changed

    def _get_related_tables(self, table_names):
        table_names = utils.unique_list(table_names)
        in_indexs = []
        out_names = []
        for name in table_names:
            if name in self._tables_index:
                in_indexs.append(self._tables_index[name])
            else:
                out_names.append(name)

        ret = set()
        for index in in_indexs:
            ret |= self._get_table_children(index)

        ret |= set(in_indexs)
        ret = list(ret)
        ret.sort()
        in_names = [self._tables_name[index] for index in ret]
        return in_names, out_names

    def _get_table_children(self, index):
        ret = set()
        for i, parents in enumerate(self._tables_parents):
            if index in parents:
                ret.add(i)
        return ret

    def get_reload_tables(self, table_names):
        indexs = []
        for name in table_names:
            if name not in self._tables_index:
                return []
            indexs.append(self._tables_index[name])

        ret = set()
        for index in indexs:
            ret |= self._get_table_children(index)

        ret |= set(indexs)
        ret = list(ret)
        ret.sort()
        ret = [self._tables_name[index] for index in ret]
        return ret

    def get_table_info(self, table_name):
        return self.run_for_json("desc table {}".format(table_name))

    def my_test(self):
        ret = self.session.catalog.listTables()
        print("listTables ret:", ret)
