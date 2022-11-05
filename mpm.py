#export_file = './ply/mpm3d.ply'  # use '/tmp/mpm3d.ply' for exporting result to disk
export_file = False
import numpy as np

import taichi as ti

ti.init(arch=ti.gpu)

dim, n_grid, steps, dt = 3, 64, 100, 1e-4


l0 = 0.3

dx = 1.0 / n_grid
n_particles_per_direction = 2
n_particles = int((l0**dim) / (dx**dim) * 8)

start_x = 0.5 - 0.5*l0
start_y = 0.6
start_z = 0.5 - 0.5*l0


p_rho = 4e2
p_vol = l0**dim / n_particles
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3

E = 1.4e5
nu = 0.2
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))
mu_0 = E / (2 * (1 + nu))

critical_compression = 2.5e-2 #1.5e-2
critical_stretch = 7.5e-3 #10e-3 
hardening_coefficient = 10.0

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F = ti.Matrix.field(dim, dim, float, n_particles)
F_J = ti.field(float, n_particles)

F_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_grid_m = ti.field(float, (n_grid, ) * dim)

neighbour = (3, ) * dim



# center ball quantities
ball_radius = 0.15
ball_center_x = 0.5
ball_center_y = 0.2
ball_center_z = 0.5


@ti.kernel
def substep():
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    # ti.block_dim(n_grid)
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        # snow
        # get trial deformation gradient
        F[p] = (ti.Matrix.identity(float, dim) + dt * F_C[p]) @ F[p]

        # hardening parameter
        h = ti.exp(hardening_coefficient * (1.0 - F_J[p]))

        la = lambda_0 * h
        mu = mu_0 * h
        
        # return mapping
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(dim)):
        	new_sig = sig[d, d]
        	new_sig = ti.min(ti.max(sig[d, d], 1 - critical_compression),
        					 1 + critical_stretch)
        	F_J[p] *= sig[d, d] / new_sig
        	sig[d, d] = new_sig
        	J *= new_sig

        	# reconstruct elastic deformation
        	F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, dim) * la * J * (J - 1)

        
        stress = (-dt * p_vol * 4 / dx / dx) * stress
        affine = ti.Matrix.identity(float, dim) * stress + p_mass * F_C[p]


        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base +
                     offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I][1] -= dt * gravity
        cond = (I < bound) & (F_grid_v[I] < 0) | \
               (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = 0 if cond else F_grid_v[I]


        # collision with the circle
        i = I[0]
        j = I[1]
        k = I[2]
        dist = ti.Vector([i*dx - ball_center_x, j*dx - ball_center_y, k*dx - ball_center_z])
        if dist.norm() < ball_radius:
            dist = dist.normalized() # now it becomes to the direction
            F_grid_v[I] -= dist * min(0, F_grid_v[I].dot(dist))



    # ti.block_dim(n_grid)
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_J[p] *= 1 + dt * new_C.trace()
        F_C[p] = new_C

count = ti.field(dtype=int, shape=())

@ti.kernel
def init():
    for i in range(n_particles):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * l0 + ti.Vector([start_x, start_y, start_z])
        F_J[i] = 1
        F[i] = ti.Matrix.identity(float, dim)




def T(a):
    if dim == 2:
        return a

    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5


def main():
    init()
    gui = ti.GUI('MPM3D', background_color=0x112F41)
    # while gui.running and not gui.get_event(gui.ESCAPE):
    for frame in range(200):
        for s in range(steps):
            substep()
        pos = F_x.to_numpy()
        if export_file:
            writer = ti.tools.PLYWriter(num_vertices=n_particles)
            writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
            writer.export_frame(gui.frame, export_file)
        gui.circles(T(pos), radius=1.5, color=0x66ccff)
        gui.show()


if __name__ == '__main__':
    main()