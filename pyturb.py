import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    u_id = 1000

    with h5py.File("x.h5", "r") as h5f:
        x = np.array(h5f["x"])
    with h5py.File("duct2200/u_" + str(u_id) + ".h5", "r") as h5f:
        u = np.array(h5f["u/" + str(u_id)])

    print "Imported it"

    ids_mean = np.all([x[:, 0] != 0., x[:, 1] != 0.], axis=0)

    u_mean = np.mean(u[ids_mean, :], 0)
    print u_mean

    len_x = np.count_nonzero(
        np.all([x[:, 1] == 0., x[:, 2] == 0.], axis=0))
    len_y = np.count_nonzero(
        np.all([x[:, 2] == 0., x[:, 0] == 0.], axis=0))
    len_z = np.count_nonzero(
        np.all([x[:, 0] == 0., x[:, 1] == 0.], axis=0))

    print len_x, len_y, len_z

    dims = (len_x, len_y, len_z)
    
    rx = x[:, 0].reshape(dims)
    ry = x[:, 1].reshape(dims)
    rz = x[:, 2].reshape(dims)

    ux = u[:, 0].reshape(dims)
    uy = u[:, 1].reshape(dims)
    uz = u[:, 2].reshape(dims)

    dux = ux-u_mean[0]
    duy = uy-u_mean[1]
    duz = uz-u_mean[2]

    coord_x = rx.mean((1, 2))
    coord_y = ry.mean((2, 0))
    coord_z = rz.mean((0, 1))

    uz_mean_xy = np.mean(uz[:, :, :], axis=2)

    duz_xy = uz[:, :, len_z/2] - uz_mean_xy

    plt.figure()
    plt.imshow(duz_xy,
               interpolation='nearest', origin='lower',
               cmap=plt.get_cmap('viridis'))

    q = np.sqrt(dux**2 + duy**2)/u_mean[2]

    dx = 2
    u_centerline = np.mean(uz[len_x/2-dx:len_x/2+dx,
                              len_y/2-dx:len_y/2+dx, :],
                           axis=(0, 1))

    q_mean_xy = np.mean(q, axis=(0, 1))

    duz_l = np.mean(
        (uz[1, :, :]-uz[0, :, :])/(coord_x[1]-coord_x[0]),
        axis=0)
    duz_r = np.mean(
        (uz[-2, :, :]-uz[-1, :, :])/(coord_x[-1]-coord_x[-2]),
        axis=0)
    duz_b = np.mean(
        (uz[:, 1, :]-uz[:, 0, :])/(coord_y[1]-coord_y[0]),
        axis=0)
    duz_t = np.mean(
        (uz[:, -2, :]-uz[:, -1, :])/(coord_y[-1]-coord_y[-2]),
        axis=0)

    duz = (duz_l + duz_r + duz_t)/3.

    plt.figure()
    plt.plot(coord_z, q_mean_xy)
    plt.plot(coord_z, u_centerline)
    plt.plot(coord_z,
             np.mean(u_centerline) * np.ones_like(u_centerline))
    plt.plot(coord_z, duz)
    
    plt.figure()
    plt.plot(coord_z, duz_l)
    plt.plot(coord_z, duz_r)
    plt.plot(coord_z, duz_t)
    plt.plot(coord_z, duz_b)
    plt.plot(coord_z, duz)
    
    plt.show()


if __name__ == "__main__":
    main()
