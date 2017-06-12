import h5py
import numpy as np
import argparse
import os


def get_tsteps(h5fu, id_first=0, id_last=-1):
    ids = sorted([int(key) for key in h5fu["VisualisationVector"].keys()])
    if id_last >= len(ids):
        id_last = len(ids)-1
    actual_ids = ids[id_first:id_last]
    actual_ids.append(ids[id_last])
    return actual_ids


def main():
    parser = argparse.ArgumentParser(description="Make an average xdmf figure")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-id_first", type=int, help="Start id", default=0)
    parser.add_argument("-id_last", type=int, help="Stop id", default=-1)
    parser.add_argument("-o", type=str, default="", help="Output folder")
    args = parser.parse_args()

    ufilename = os.path.join(args.folder, "u_from_tstep_0.h5")
    pfilename = os.path.join(args.folder, "p_from_tstep_0.h5")
    if not os.path.isfile(ufilename):
        exit("Couldn't find " + ufilename)
    if not os.path.isfile(pfilename):
        exit("Couldn't find " + pfilename)

    id_first = args.id_first
    id_last = args.id_last
    with h5py.File(ufilename, "r") as h5fu:
        tsteps = get_tsteps(h5fu, id_first, id_last)
        n_tsteps = len(tsteps)
        print "Timesteps:", tsteps, "(" + str(n_tsteps) + " timesteps)"

        u_avg = np.zeros_like(h5fu["VisualisationVector"][str(tsteps[0])])
        for tstep in tsteps:
            print tstep
            u_avg[:, :] += np.array(
                h5fu["VisualisationVector"][str(tstep)])/n_tsteps

        elem = np.array(h5fu["Mesh/0/mesh/topology"])
        node = np.array(h5fu["Mesh/0/mesh/geometry"])

    with h5py.File(pfilename, "r") as h5fp:
        tsteps = get_tsteps(h5fp, id_first, id_last)
        n_tsteps = len(tsteps)
        print "Timesteps:", tsteps, "(" + str(n_tsteps) + " timesteps)"

        p_avg = np.zeros_like(h5fp["VisualisationVector"][str(tsteps[0])])
        for tstep in tsteps:
            print tstep
            p_avg[:] += np.array(
                h5fp["VisualisationVector"][str(tstep)])/n_tsteps

    if args.o == "":
        u_avg_file_h5 = args.folder + "/u_avg.h5"
        file_xdmf = args.folder + "/visualise.xdmf"
        p_avg_file_h5 = args.folder + "/p_avg.h5"
    else:
        if not os.path.exists(args.o):
            os.makedirs(args.o)
        u_avg_file_h5 = os.path.join(args.o, "u_from_tstep_0.h5")
        p_avg_file_h5 = os.path.join(args.o, "p_from_tstep_0.h5")

        file_xdmf = os.path.join(args.o, "visualise.xdmf")

    with h5py.File(u_avg_file_h5, "w") as h5fu_out:
        h5fu_out.create_dataset("Mesh/0/mesh/geometry", data=node)
        h5fu_out.create_dataset("Mesh/0/mesh/topology", data=elem)
        h5fu_out.create_dataset("VisualisationVector/0", data=u_avg)

    with h5py.File(p_avg_file_h5, "w") as h5fp_out:
        h5fp_out.create_dataset("VisualisationVector/0", data=p_avg)

    # h5fdu_out = h5py.File(args.folder + "/u_dev.h5", "w")
    # with h5py.File(ufilename, "r") as h5fu:
    #     for tstep in tsteps:
    #         h5fdu_out.create_dataset("data/

    # Saved...
    # <Geometry GeometryType="XYZ">
    #      <DataItem Dimensions="{n_node} 3" Format="HDF">u_from_tstep_0.h5:/Mesh/0/mesh/geometry</DataItem>
    #    </Geometry>

    xdmf_str = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Time TimeType="List">
        <DataItem Format="XML" Dimensions="1"> 0</DataItem>
      </Time>
      <Grid Name="mesh" GridType="Uniform">
        <Topology NumberOfElements="{n_elem}" TopologyType="Tetrahedron" NodesPerElement="4">
          <DataItem Dimensions="{n_elem} 4" NumberType="UInt" Format="HDF">{u_avg_h5}:/Mesh/0/mesh/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{n_node} 3" Format="HDF">{u_avg_h5}:/Mesh/0/mesh/geometry</DataItem>
        </Geometry>
        <Attribute Name="Assigned Vector Function" AttributeType="Vector" Center="Node">
          <DataItem Dimensions="{n_node} 3" Format="HDF">{u_avg_h5}:/VisualisationVector/0</DataItem>
        </Attribute>
        <Attribute Name="p" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="{n_node} 1" Format="HDF">{p_avg_h5}:/VisualisationVector/0</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>""".format(n_elem=len(elem), n_node=len(node),
                  u_avg_h5=u_avg_file_h5.split("/")[-1],
                  p_avg_h5=p_avg_file_h5.split("/")[-1])

    with file(file_xdmf, "w") as xdmff_out:
        xdmff_out.write(xdmf_str)

if __name__ == "__main__":
    main()
