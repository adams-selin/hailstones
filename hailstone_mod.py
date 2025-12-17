import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import trimesh
from copy import deepcopy
import pymeshfix as mf

def recalculate_normals_from_centroid(mesh):
    """
    Recalculate all normals to point outward from the mesh centroid.
    Handles triangular, quad, and mixed polygon meshes.

    Parameters
    ----------
    mesh : pv.PolyData
        The mesh to process

    Returns
    -------
    mesh : pv.PolyData
        The mesh with corrected normals
    """
    # Get the centroid of the entire mesh
    centroid = mesh.center

    # Get cell centers
    cell_centers = mesh.cell_centers().points

    # Calculate vectors from centroid to each cell center
    outward_vectors = cell_centers - centroid

    # Normalize these vectors to get desired normal directions
    desired_normals = outward_vectors / np.linalg.norm(outward_vectors, axis=1, keepdims=True)

    # Compute the current normals
    mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    current_normals = mesh.cell_data['Normals']

    # Check which normals need flipping
    dot_products = np.sum(current_normals * desired_normals, axis=1)
    needs_flipping = dot_products < 0

    # Process faces - need to handle variable polygon sizes
    faces = mesh.faces.copy()
    n_cells = mesh.n_cells

    # Parse faces array and flip as needed
    idx = 0
    for cell_id in range(n_cells):
        # First value is number of points in this face
        n_points = faces[idx]

        # Get the vertex indices for this face
        vertex_indices = faces[idx+1:idx+1+n_points]

        # If this face needs flipping, reverse the vertex order
        if needs_flipping[cell_id]:
            faces[idx+1:idx+1+n_points] = vertex_indices[::-1]

        # Move to next face
        idx += n_points + 1

    # Update the mesh faces
    mesh.faces = faces

    # Recompute normals after flipping
    mesh.compute_normals(cell_normals=True, point_normals=True, consistent_normals=True)

    print(f"Flipped {needs_flipping.sum()} out of {len(needs_flipping)} faces")

    return mesh




def fill_all_hole_loops(mesh, holes):
    """
    Fill all hole loops on a triangulated PyVista mesh by creating fan triangles
    from each loop's centroid to its boundary vertices.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Triangulated input mesh (will not be modified in-place).
    holes : pyvista.PolyData
        Lines polydata returned by MeshFix.extract_holes() (edge format).

    Returns
    -------
    pyvista.PolyData
        New mesh with all holes filled (triangles appended).
    """
    import numpy as np
    import pyvista as pv

    if holes is None or holes.n_points == 0 or holes.n_cells == 0:
        return mesh.copy()

    # Parse holes.lines to extract edges (2-point line segments)
    lines = holes.lines.astype(int)
    edges = []
    idx = 0
    while idx < len(lines):
        n = int(lines[idx])
        if n == 2:  # edge
            p0, p1 = lines[idx+1], lines[idx+2]
            edges.append((p0, p1))
        idx += n + 1

    if not edges:
        return mesh.copy()

    # Reconstruct closed loops from edges
    # Build adjacency: for each point, what points does it connect to?
    from collections import defaultdict
    adj = defaultdict(list)
    for p0, p1 in edges:
        adj[p0].append(p1)
        adj[p1].append(p0)

    # Extract loops by following edges
    loops = []
    visited_edges = set()

    for start_p, neighbors in adj.items():
        for next_p in neighbors:
            edge_key = (min(start_p, next_p), max(start_p, next_p))
            if edge_key in visited_edges:
                continue

            # Trace loop starting from this edge
            loop = [start_p, next_p]
            current = next_p
            prev = start_p
            visited_edges.add(edge_key)

            while current != start_p:
                # Find next neighbor (not the one we came from)
                next_neighbors = [p for p in adj[current] if p != prev]
                if not next_neighbors:
                    break  # Dead end, not a closed loop

                next_p = next_neighbors[0]
                edge_key = (min(current, next_p), max(current, next_p))
                visited_edges.add(edge_key)

                if next_p == start_p:
                    # Loop closed
                    if len(loop) >= 3:
                        loops.append(np.array(loop))
                    break
                else:
                    loop.append(next_p)
                    prev = current
                    current = next_p

    if not loops:
        print("No closed loops found in hole edges")
        return mesh.copy()

    print(f"Found {len(loops)} hole loops")

    # Copy original mesh data
    orig_pts = mesh.points.copy()
    orig_faces = mesh.faces.astype(int)

    # Convert existing faces to explicit triangle list. Not necessary if already triangulated.
    tris = []
    j = 0
    while j < len(orig_faces):
        nv = int(orig_faces[j])
        verts = orig_faces[j + 1: j + 1 + nv].tolist()
        if nv == 3:
            tris.append(verts)
        else:
            # fan-triangulate polygon
            for k in range(1, nv - 1):
                tris.append([verts[0], verts[k], verts[k + 1]])
        j += nv + 1

    tris = np.array(tris, dtype=int)

    # Start with original mesh points, then append hole boundary points and centroids
    new_points = orig_pts.tolist()
    new_tris = tris.tolist()

    # Map from holes.points indices to new_points indices
    hole_pts_offset = len(new_points)
    for pt in holes.points:
        new_points.append(pt.tolist())

    # Fill each hole loop with fan triangles to its centroid
    for loop_idx, loop in enumerate(loops):
        # loop contains indices into holes.points
        # Map them to indices in new_points
        loop_indices_in_new = [hole_pts_offset + idx for idx in loop]

        # Compute centroid from actual hole boundary points
        coords = holes.points[loop]
        centroid = coords.mean(axis=0)
        centroid_idx = len(new_points)
        new_points.append(centroid.tolist())

        # Create fan triangles: (loop[i], loop[i+1], centroid) for each edge
        L = len(loop_indices_in_new)

        for k in range(L):
            i0 = loop_indices_in_new[k]
            i1 = loop_indices_in_new[(k + 1) % L]
            new_tris.append([i0, i1, centroid_idx])

    # Build VTK faces array from triangles: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    new_tris = np.array(new_tris, dtype=np.int64)

    # Create faces array: prepend count (3) to each triangle
    faces_list = []
    for tri in new_tris:
        faces_list.append(3)
        faces_list.extend(tri)

    faces_vtk = np.array(faces_list, dtype=np.int64)

    # Create and return new mesh
    new_pts_arr = np.array(new_points, dtype=float)
    print(f"Created mesh with {len(new_tris)} triangles, {len(new_pts_arr)} points")
    new_mesh = pv.PolyData(new_pts_arr, faces_vtk)

    # Recompute normals
    new_mesh.compute_normals(cell_normals=True, point_normals=True, consistent_normals=True)

    return new_mesh





###################BEGIN PROGRAM###################
#Number of lobes you want on your hailstone.
n_lobes = 100

#How narrow you want your lobes to be. Decimal number between
# 0 (lobe nonexistent, no point!). to 0.45 (as large as possible
# without overlap) to something higher (at 1.0, they will fully
# overlap with the neighboring node). Note lobe width is also
# dependent on the number of lobes, and the sine/cosine equation
# used to generate them. That equation would need to be modified
# in code below.
lobe_factor = 0.75

#Lobe height factor. Setting it to 1 will make the lobes a factor
# of 1.5*radius. Setting it to 0 will make the lobes the same
# level as the radius (non-existent).
height_factor = 0.25

#Radius of the sphere underlying the lobes.
hail_radius = 60. #mm

# Create a range of equidistant points on a sphere using Deserno 2004,
# but with correction that points are first determined on unit sphere, 
# then expanded by conversion to cartesian coordinates

#Note - we always start with a unit sphere, and only expand to the 
# selected radius after all points/lobes are generated.

#Number of equidistant point to generate around the sphere 
# to use to make the spherical mesh. 5000 is good for 50 lobes.
N = 4000 ##################<--Can this be reduced??
r = 1 #radius in mm
sphere_points=[]
n_count = 0 # actual number of points around sphere


#a = 4*np.pi*r**2./ N
a = 4*np.pi / N
d = np.sqrt(a)
M_theta = np.round(np.pi/d)
d_theta = np.pi/M_theta
d_phi = a/d_theta
for m in np.arange(0,M_theta):
    theta = np.pi*(m+0.5) / M_theta
    M_phi = np.round(2*np.pi*np.sin(theta)/d_phi)
    for n in np.arange(0,M_phi):
        phi = 2*np.pi*n/M_phi
        x = r * np.sin(theta)*np.cos(phi)
        y = r * np.sin(theta)*np.sin(phi)
        z = r * np.cos(theta)
        sphere_points.append((x,y,z))
        n_count += 1

print (n_count)
sphere_points = np.array(sphere_points)





lobe_centers = [] #just the center points
lobes = [] #successive sets of points, including circles around center points, one for each lobe
lobe_count = 0 #actual number of lobes
lobe_thetas = []
lobe_phis = []

#get the spherical coordinates (on a unit sphere) of all lobe centers
#basically, repeat the stuff in previous cell but for fewer points
a = 4*np.pi / n_lobes
d = np.sqrt(a)
M_theta = np.round(np.pi/d)
#d_theta = np.pi/M_theta
d_theta = 0.95*np.pi/M_theta #seemed to crowd at the poles, so this pulls it away a bit more
d_phi = a/d_theta
for m in np.arange(0,M_theta):
    #theta = np.pi*(m+0.5) / M_theta
    theta = 0.95*np.pi*(m+0.55) / M_theta
    #theta = np.pi*(m+0.55) / M_theta
    M_phi = np.round(2*np.pi*np.sin(theta)/d_phi)
    for n in np.arange(0,M_phi):
        phi = 2*np.pi*n/M_phi
        lobe_thetas.append(theta)
        lobe_phis.append(phi)
        x = r * np.sin(theta)*np.cos(phi)
        y = r * np.sin(theta)*np.sin(phi)
        z = r * np.cos(theta)
        lobe_centers.append((x,y,z))
        lobe_count += 1

#now calculate circles around each lobe point, at a central angle of alpha
# following https://math.stackexchange.com/questions/643130/circle-on-sphere
# Ah ha! But note their phi and theta angles are switched from the Deserno 2004 paper
alpha_start = lobe_factor*np.minimum(d_theta, d_phi)
radius = 1. #set to unit sphere radius to start
num_t_at_start = 40
circ_at_start = 2*np.pi*alpha_start
dt = circ_at_start / num_t_at_start
#circles per lobe
radial_steps = 20
#points per circle
circle_resolution = 30
#gives me 20*30 = 600 points per lobe...so nlobes=100 * 600 = 60k points


for phi, theta, lobe_center in zip(lobe_phis, lobe_thetas, lobe_centers):
    beta = theta
    gamma = np.pi - phi
    lobe_points = []
    
    #cycle through successively smaller alphas, setting the radius/height increasingly higher
    # each time
    #alphas = np.linspace(alpha_start, 0, num_t_at_start)
    alphas = np.linspace(alpha_start, 0, radial_steps)
    #lobe_heights = radius + np.sin(np.pi/2 * np.linspace(0,0.5,10)) #equation too sharp
    #lobe_heights = radius + height_factor*\
    #                (-(np.cos(np.pi*np.linspace(0,1.0,num_t_at_start))-1)/4)  #followed https://easings.net/#easeInOutSine
    lobe_heights = radius + height_factor*(-(np.cos(np.pi*np.linspace(0,1.0,radial_steps))-1)/4)
    first_time_through = True
    for alpha, lobe_height in zip(alphas[:-1], lobe_heights[:-1]):
    
        #calculate number of t needed for this alpha
        #circ_here = 2*np.pi*alpha
        #num_t = int(np.round(circ_here/dt))
        #t = np.linspace(0,2*np.pi,num_t)
        t = np.linspace(0,2*np.pi, circle_resolution)
        #print (alpha, circ_here, num_t)

        #cartesian coordinates for each point in the circle 
        xs = lobe_height*(np.sin(alpha)*np.cos(beta)*np.cos(gamma)*np.cos(t) + 
                            np.sin(alpha)*np.sin(gamma)*np.sin(t)-
                            np.cos(alpha)*np.sin(beta)*np.cos(gamma))
        ys = lobe_height*(-np.sin(alpha)*np.cos(beta)*np.sin(gamma)*np.cos(t) + 
                            np.sin(alpha)*np.cos(gamma)*np.sin(t)+
                            np.cos(alpha)*np.sin(beta)*np.sin(gamma))
        zs = lobe_height*(np.sin(alpha)*np.sin(beta)*np.cos(t)+np.cos(alpha)*np.cos(beta))
        
        # if first time through, calculate great circle distance from center point to all points in circle
        # (should all be the same)
        if first_time_through:
            #gcd between circle and center point on unit sphere is just the angle between them - alpha!
            #cartesian coordinates calculation if you want to prove it, below
            # gcd = []
            # for thisx,thisy,thisz in zip(xs,ys,zs):
            #     a = np.array((thisx,thisy,thisz))
            #     b = np.array(lobe_centers[0])
            #     a_dot_b = np.dot(a,b)
            #     angle = np.arccos(a_dot_b / ( (a[0]**2+a[1]**2+a[2]**2)**0.5 * (b[0]**2+b[1]**2+b[2]**2)**0.5 ) )
            #     gcd.append(r*angle)
            gcd = alpha
            
            #now, determine which points in sphere points array are closer to center point than that.
            #remove them, as they will become "inside" points underneath the lobe once surfaces are merged.
            a = sphere_points
            b = np.array(lobe_center)
            a_dot_b = np.dot(a,b)
            angle = np.arccos(a_dot_b / ( (a[:,0]**2+a[:,1]**2+a[:,2]**2)**0.5 * (b[0]**2+b[1]**2+b[2]**2)**0.5 ) )
            points_gcds = r*angle
            inside_circle = points_gcds < gcd
            sphere_points = np.delete(sphere_points,inside_circle,axis=0)

        #last point is right at alpha = 0
        alpha = alphas[-1]
        lobe_height = lobe_heights[-1]
        xs = np.append(xs, lobe_height*(np.sin(theta)*np.cos(phi))) #use the x,y,z location of lobe center but add lobe_height
        ys = np.append(ys, lobe_height*(np.sin(theta)*np.sin(phi)))
        zs = np.append(zs, lobe_height*(np.cos(theta)))
        
        dum = [lobe_points.append((i,j,k)) for i,j,k in zip(xs,ys,zs)]
        first_time_through = False
    
    lobes.append(lobe_points)

print ('len of lobes', len(lobes))
print ('len of lobe centers', len(lobe_centers))
print ('sphere_points.shape', sphere_points.shape)

          

#expand sphere points to requested radius,
# and convert to point cloud
#sphere_cloud = pv.PolyData(hail_radius*sphere_points) #dont need now
#calculate a separate surface for each lobe
lobe_surfs = []
#all_lobe_points = []
for lobe in lobes:
    #all_lobe_points = all_lobe_points+lobe
    lobe_points = np.array(lobe)
    #expand lobe points to requested radius
    lobe_point_cloud = pv.PolyData(hail_radius*lobe_points)
    #lobe_surfs.append(lobe_point_cloud.reconstruct_surface(sample_spacing=0.05)) #VERY MEMORY HUNGRY!
    #triangulate our lobes instead
    lobe_surf = lobe_point_cloud.delaunay_2d()
    lobe_surfs.append(lobe_surf)
print('done with lobe for loop')
#add our surfaces together
#sphere_surf = sphere_cloud.reconstruct_surface() #VERY MEMORY HUNGRY!
#construct sphere naturally...increase theta, phi will result in larger stl file
#so 100,100 will be smoother sphere, more triangles compared to smaller
#keep same
#https://docs.pyvista.org/api/utilities/_autosummary/pyvista.sphere
sphere_surf = pv.Sphere(radius=hail_radius, theta_resolution=100, phi_resolution=100)
print('sphere surf done')
total_surf = deepcopy(sphere_surf)
for lobe_surf in lobe_surfs:
    #total_surf = total_surf + lobe_surf 
    #gives me fewer duplicates points to clean
    total_surf = total_surf.merge(lobe_surf)
#convert polygons to triangles
total_surf = total_surf.triangulate()

# Fix the surface normals to make them all point outward
cell_normals_recal = recalculate_normals_from_centroid(total_surf)

# #and plot, by individual lobe and then combined surface    
pv.global_theme.color_cycler = 'default'
pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
_ = pl.add_mesh(sphere_surf)
#_ = pl.add_mesh(lobe_point_cloud,color='k')
for lobe_surf in lobe_surfs:
    _ = pl.add_mesh(lobe_surf)
    
pl.subplot(0, 1)
_ = pl.add_mesh(total_surf,color='blue')
pl.show()




#save data
total_surf.save(str(hail_radius)+'mm_equimesh_n'+str(int(n_lobes))+'lobe_w'+str(int(lobe_factor*100))+
                '_h'+str(int(height_factor*100))+'.stl')

#now plot again
mesh = trimesh.load_mesh(str(hail_radius)+'mm_equimesh_n'+str(int(n_lobes))+'lobe_w'+str(int(lobe_factor*100))+
                '_h'+str(int(height_factor*100))+'.stl')
# Show the mesh (opens in a window)
mesh.show()



#mesh quality
print(f"- Is watertight: {mesh.is_watertight}")
print(f"- Volume: {mesh.volume:.4f}")
print(f"- Surface area: {mesh.area:.4f}")









tri_surf = total_surf.triangulate()
mesh_to_fix = mf.MeshFix(tri_surf)
holes = mesh_to_fix.extract_holes()

# Show holes before filling
p = pv.Plotter()
p.add_mesh(tri_surf, color='blue')
p.add_mesh(holes, color='red', line_width=8)
p.show()

# Fill all holes
filled = fill_all_hole_loops(tri_surf, holes)
print(f"Original mesh: {tri_surf.n_points} points, {tri_surf.n_cells} cells")
print(f"Filled mesh: {filled.n_points} points, {filled.n_cells} cells")

# Check if watertight (convert to trimesh to access is_watertight)
filled_trimesh = trimesh.Trimesh(vertices=filled.points, faces=filled.faces.reshape(-1, 4)[:, 1:])
print(f"Is filled mesh watertight? {filled_trimesh.is_watertight}")

# Visualize result
#filled.plot()







# Diagnostic: analyze what holes remain
print("\n=== DIAGNOSTIC ===")
print(f"Original mesh is_watertight: {mesh.is_watertight}")
print(f"Filled mesh (trimesh) is_watertight: {filled_trimesh.is_watertight}")
print(f"Filled mesh volume: {filled_trimesh.volume:.6f}")

# Check if there are still holes in the filled mesh
mesh_to_fix_after = mf.MeshFix(filled)
holes_after = mesh_to_fix_after.extract_holes()

print(f"\nHoles before filling: {holes.n_cells} hole segments")
print(f"Holes after filling: {holes_after.n_cells} hole segments")

# The key question: is it actually watertight for practical purposes?
if filled_trimesh.is_watertight:
    print("\n✓ SUCCESS: Mesh is watertight according to trimesh!")
    print("  (The remaining holes detected by MeshFix are likely topological artifacts)")
    print("  Your mesh is ready for use.")

    # Save the watertight mesh
    filled_trimesh.export(str(hail_radius)+'mm_equimesh_n'+str(int(n_lobes))+'lobe_w'+str(int(lobe_factor*100))+
                '_h'+str(int(height_factor*100))+'_watertight.stl')
    print('  Saved to: '+str(hail_radius)+'mm_equimesh_n'+str(int(n_lobes))+'lobe_w'+str(int(lobe_factor*100))+
                '_h'+str(int(height_factor*100))+'_watertight.stl')
else:
    print("\n✗ Mesh is still not watertight")
    # Visualize remaining holes if any
    if holes_after.n_cells > 0:
        print("Remaining holes detected! Visualizing...")
        p2 = pv.Plotter()
        p2.add_mesh(filled, color='blue', opacity=0.7)
        p2.add_mesh(holes_after, color='red', line_width=10)
        p2.show()
