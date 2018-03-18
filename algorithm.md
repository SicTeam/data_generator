## Modified Algorithm Preparing a Data-Set: ##
 1. S ← predefined size intervals
 2. D ← foregrounds of drone images
 3. B ← foregrounds of bird images
 4. V ← background videos
 5. R ←  # of rows that the image will be divided into
 6. C ←  # of columns that the image will be divided into
 7. G ← R × C grid
 8. foreach(d, g, s, v) ∈ D × G × S × V do
 9.    ignore this configuration with probability:
          p = 1 − Max.allowed size / Total size for all configurations
       and continue
10.    draw a random position p0 in g
11.    draw a random size s0 for smaller edge of the drone from s
12.    draw a random frame f0 from v
13.    resize d with respect to s0
14.    overlay f0 with d in position p0
15.    draw (p1, s1, f1) in the same way
16.    resize d with respect to s1
17.    overlay f1 with d in position p1
18.    draw (p2, s2, f2) in the same way
19.    resize d with respect to s2
20.    overlay f2 with d in position p2
21.    draw a random bird b0 from B and a random position
22.    draw (pb,0, sb,0) for bird b0
             where pb,0 is a random grid position from grid list
             where sb,0 is drawn from smaller half of S
23.    resize b0 with respect to sb,0
24.    draw a random bird b1 from B and random position
25.    draw (pb,1, sb,1) for bird b1
               where pb,1 is a random grid position from grid list
              where sb,1 is drawn from greater half of S
26.    resize b1 with respect to sb,1
27.    overlay f1 with b0 in position pb,0
28.    randomly decide to overlay a second bird
29.    overlay f1 with b1 in position pb,1
30.    grab random frame from v f3
31.    overlay f3 with b0 in position pb,0
32.    randomly choose to add a second bird
33.    overlay f3 with b1 in position pb,1
34.    save f0, f1, f2, f3 into the data set
35. End Loop
