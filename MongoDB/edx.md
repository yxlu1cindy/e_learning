Compass: MongoDB GUI

a database serves as a namespace for collections; Collections store individual records called documents

documents - one record -> can be nested -> help model the data
Collections - table

schema - > have a overall view of the data


- Filtering:
   - {'end station name': 'W 21 St & 6 Ave','birth year': {$gte: 1985,$lt: 1990}}   : JSON Type:gte-greater or equal than; lt:less than
   - Geospatial filtering: {coordinates: {$geoWithin: { $centerSphere: [ [ -71.88737045912076, 21.754950558720125 ], 0.14906848415689494 ]}}}   or hold shift and select the region;0.149 is the radius
