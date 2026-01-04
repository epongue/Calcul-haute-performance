Mesh.MshFileVersion = 2.2;
Mesh.CharacteristicLengthFactor = 0.5;
Mesh.CharacteristicLengthMax = 1;
Mesh.Optimize = 1;

Point(1) = {-1,-1,-1};
Point(2) = {-1, 1,-1};
Point(3) = { 0,-1,-1};
Point(4) = { 0, 1,-1};
Point(5) = { 1,-1,-1};
Point(6) = { 1, 1,-1};
Line(1) = {1,2};
Line(2) = {3,4};
Line(3) = {5,6};
Line(5) = {1,3};
Line(6) = {3,5};
Line(7) = {2,4};
Line(8) = {4,6};
Line Loop(1) = {1,7,-2,-5};
Line Loop(2) = {2,8,-3,-6};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
tmp[] = Extrude {0,0,2} {
  Surface{1};
  Surface{2};
};
Physical Volume(100) = {1};
Physical Volume(200) = {2};
Physical Surface(300) = {52,51,47,43,30,29,25,21,17,2,1};
