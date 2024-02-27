# Gaussian_Point_Matrix
This repo implements the matrix to measure the difference between Gaussian Splat and Point Cloud 

## Matrix
- Preserving Score
- Clearance Score
- Coherent Score

## Details
- we have over all three matrix to describe how close the point cloud and Gaussian Splatting. They are Preservation, Clearance, Coherent Score
- If a point is contained in a Gaussian Splat, we call there is a match between point and Gaussian Splat.
- the ratio between All points that has a match to the number of all points is the preservation score
- the ratio between all Gaussian Splat that has a match to the number of all Gaussian Splat is the clearance score
- When there is a matching, if the color between Gaussian and the point is close, then we say we have a coherent matching point. The number of coherent matching points over the number of matching points is the coherent score.
