Compass: MongoDB GUI <- visual based
Mongo Shell: Text based <- full support for query language

a database serves as a namespace for collections; Collections store individual records called documents

documents - one record -> can be nested -> help model the data
Collections - table

schema - > have a overall view of the data

Compass:
- Filtering:
   - {'end station name': 'W 21 St & 6 Ave','birth year': {$gte: 1985,$lt: 1990}}   : JSON Type:gte-greater or equal than; lt:less than
   - Geospatial filtering: {coordinates: {$geoWithin: { $centerSphere: [ [ -71.88737045912076, 21.754950558720125 ], 0.14906848415689494 ]}}}   or hold shift and select the region;0.149 is the radius
   

- CRUD (Create, Read, Update, Delete)

Shell:
In Commander
mongo "mongodb://cluster0-shard-00-00-jxeqq.mongodb.net:27017,cluster0-shard-00-01-jxeqq.mongodb.net:27017,cluster0-shard-00-02-jxeqq.mongodb.net:27017/test?replicaSet=Cluster0-shard-0" --authenticationDatabase admin --ssl --username m001-student --password m001-mongodb-basics

> use video ->switch database    
> show collwctions -> show all tables under the database    
> db.movies.find({"mpaaRating":"PG-13"}).pretty() -> show records in the movie table by mpaaRating filters    
> db.movies.find({"mpaaRating":"PG-13"}).count() -> counts    
> db.movies.find({"mpaaRating":"PG-13"},{title:1, \_id=0}) -> only show title for each records  1-show;0-not show
- Load Data
   S1: set dir to the file location
   S2: load("filename")

- Insert Document:
   db.movieScratch(collection name).insertOne({c1:"",...})

- Update Documents ->upsert:update if exists, or insert:
   db.movieScratch(collection name).updateOne({c1:"",...},{$set:{c2:"xxx"}})   the first {} is filter the records we want to update,{c2:} means the columns we want to modify
