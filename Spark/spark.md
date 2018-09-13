1. Hadoop:
  - a way to distribute large file on multiple machine
  - uses HDFS
  - duplicate sevral block of data for fault tolerance(one machine go down doesn't matter)
  - map & reduce -distribut computational task

2. Spark:
  - a flexible alternative for MapReduce
  - can use data stored at AWS3, HDFS, cASSENDRA..

3. SPARK vs MapReduce:
  - data format
  - fast speed
  - MR writes most data to disk after each map$reduce operation; spark keeps data in memory after each operation
  
4. Spark RDD:
  - pros :parallel, dault tolerent, distribution, ability to use many data source
  
5. Type of Spark operation:
  - Transformations
  - Action
