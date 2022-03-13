import numpy as np

class Marcher:

    def __init__(self, bottom_left = None, top_right = None, volumefilepath = None):
        if (bottom_left is None) or (top_right is None) or (volumefilepath is None):
            return
        self.lowest = np.array([0,0,0])
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.volume = np.load(volumefilepath)
        self.voxel_counts = np.shape(self.volume)
        self.voxel_counts = np.array([self.voxel_counts[0], self.voxel_counts[1], self.voxel_counts[2]])
        self.voxel_size = (self.top_right - self.bottom_left)/self.voxel_counts
        pass

    def get_containing_voxel(self, point, direction):
        #calculate the steps for the direction
        steps = (direction < 0) * -2 + 1

        #claculate t values of direction to produce each of the voxel dimension sizes
        deltas = abs(np.divide(self.voxel_size,direction))

        #find the voxel coord of the voxel that contains the point
        vox_coord = np.floor((point-self.bottom_left)/ self.voxel_size)

        point_modulo = point % self.voxel_size

        #find the maximum t vlues for the direction that can be moved before crossing a voxel
        tmaxs = direction * 0
        tmaxs[(direction < 0)] = point_modulo[(direction < 0)]/direction[(direction<0)]
        tmaxs[(direction > 0)] = (self.voxel_size[(direction > 0)] - point_modulo[(direction > 0)])/direction[(direction>0)]
        tmaxs= np.abs(tmaxs)

        return vox_coord, steps, deltas, tmaxs

    def trace_no_scaling(self, point, direction):
        index, steps, deltas, tmaxs = self.get_containing_voxel(point, direction)
        out_of_range_count = 0
        total = 0
        while True:
            if not (np.any(index >= self.voxel_counts) or np.any(index < self.lowest)):
                total = total + self.volume[int(index[0])][int(index[1])][int(index[2])]
            if tmaxs[0] < tmaxs[1]:
                if tmaxs[0] < tmaxs[2]:
                    index[0] = index[0] + steps[0]
                    tmaxs[0] = tmaxs[0] + deltas[0]
                else:
                    index[2] = index[2] + steps[2]
                    tmaxs[2] = tmaxs[2] + deltas[2]
            else:
                if tmaxs[1] < tmaxs[2]:
                    index[1] = index[1] + steps[1]
                    tmaxs[1] = tmaxs[1] + deltas[1]
                else:
                    index[2] = index[2] + steps[2]
                    tmaxs[2] = tmaxs[2] + deltas[2]
            if np.any(index >= self.voxel_counts) or np.any(index < self.lowest):
                out_of_range_count = out_of_range_count + 1
            if out_of_range_count == 4:
                break
        return total

    def trace_scaling(self, point, direction):
        index, steps, deltas, tmaxs = self.get_containing_voxel(point, direction)
        out_of_range_count = 0
        total = 0
        while True:
            dim = 0
            #find the axis that takes the lowest t value to cross into a different voxel in
            if tmaxs[0] < tmaxs[1]:
                if tmaxs[0] < tmaxs[2]:
                    dim = 0
                else:
                    dim = 2
            else:
                if tmaxs[1] < tmaxs[2]:
                    dim = 1
                else:
                    dim = 2

            #if the voxel just traversed was in the volume, add the density of that voxel scaled by the distance travelled in that volume
            if not(np.any(index >= self.voxel_counts) or np.any(index < self.lowest)):
                total = total + tmaxs[dim] * self.volume[int(index[0])][int(index[1])][int(index[2])]

            #increment the voxel coordinate
            index[dim] = index[dim] + steps[dim]
            #update the tvalues to reflect the movement along the ray
            tmaxs = tmaxs - tmaxs[dim]
            #the tvalue dimension for the value just crossed should now be reset to the total value required to cross that axis of a voxel
            tmaxs[dim] = deltas[dim]

            #if the voxel is out of range 4 times, then stop traversing, this takes care of when the ray starts just outside of the volume
            #this theoretically doesn't happen, but due to our lovely friend floating point error is possible
            #this also takes care of when the ray starts at any of the positive boundaries of the volume, the best example is the top right corner
            #(the voxel it will be in will be self.voxel_counts) and it will take 2-3 traverses to move into the volume
            if np.any(index >= self.voxel_counts) or np.any(index < self.lowest):
                out_of_range_count = out_of_range_count + 1
            if out_of_range_count == 4:
                break
        return total
