# PROMPTS.md

1. I have a task to build point cloud localizer using icp algorithm implemented from scartch and i am not allowed to call any icp library function first explain me the algorithm in detail and tell me how to implement it.

2. why we using svd of matrix to find rotation and translation in the implemenation of icp. like what me whats the intuition of using it 

3. my rotation matrix is coming out with determinant -1 sometimes instead of +1. i know this means it is a reflection not a rotation
but i dont understand why SVD gives this and how to fix it

4. why we add noise to both source and target separately when generating the synthetic pair. why not just add it to one. what difference does it make to the icp result

5. What is geodesic rotation error and why it is better than just comparing rotation matrices element by element.

6. I want to write pytest tests that verify if my icp recovers a known transform to within 1 degree rotation and 1cm translation
on noise-free data or not tell me how to structure these tests.


