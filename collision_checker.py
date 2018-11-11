#!/usr/bin/env python
import numpy as np


class CollisionChecker():
    def __init__(self, world, delta=0.01, verbose=False):
        if verbose:
            print('\n[ Collision Checker instance setup ]')
        self.objects = world.objects
        self.frame = world.frame
        self.object_type = world.object_type

        num_grid = np.shape(world.objects)
        self.frame_max = np.max(world.frame, axis=0)
        self.frame_min = np.min(world.frame, axis=0)
        self.delta = delta

        if self.object_type == 'poly':
            if verbose:
                print('- Object type: polygon')
            self.point_validation = self.point_validation_for_poly
            self.line_validation = self.line_validation_for_poly

        elif self.object_type == 'grid':
            if verbose:
                print('- Object type: grid')
            self.grid_delta = (self.frame_max - self.frame_min) / num_grid
            self.point_validation = self.point_validation_for_grid
            self.line_validation = self.line_validation_for_grid

        else:
            print('** Warning: object_type is not defined.')

    def crossing_number_algorithm(self, pt, poly):
        '''
        Crossing Number Algorithm
        https://www.nttpc.co.jp/technology/number_algorithm.html
        '''

        cn = 0
        poly = np.array(poly)
        for i in range(len(poly) - 1):
            if ((poly[i, 1] <= pt[1])
                    and (poly[i + 1, 1] > pt[1])) \
                    or ((poly[i, 1] > pt[1])
                        and (poly[i + 1, 1] <= pt[1])):

                xd = poly[i, 0] \
                    + (pt[1] - poly[i, 1]) \
                    / (poly[i + 1, 1] - poly[i, 1]) \
                    * (poly[i + 1, 0] - poly[i, 0])

                if pt[0] < xd:
                    cn = cn + 1

        if cn % 2 == 0:
            is_outer = True
        else:
            is_outer = False
        return is_outer

    def point_validation_for_poly(self, pt):
        # For objects
        list = [self.crossing_number_algorithm(pt, obj)
                for obj in self.objects]

        # For frame
        if self.frame is not None:
            out_of_frame = self.crossing_number_algorithm(pt, self.frame)
        else:
            out_of_frame = False

        return all(list) and not out_of_frame

    def point_validation_for_grid(self, pt):

        # For frame
        if (self.frame_min[0] > pt[0] or pt[0] > self.frame_max[0] or
                self.frame_min[1] > pt[1] or pt[1] > self.frame_max[1]):
            return False

        # For objects
        index = ((pt - self.frame_min) / self.grid_delta).astype('int32')
        is_valid = bool(self.objects[index[0]][index[1]])

        return is_valid

    def line_validation_for_poly(self, p0, p1):

        # Narrow down valid objects
        valid_objects = []
        for obj in self.objects:
            tmp = [max(p0[i], p1[i]) > np.min(obj[:, i]) and
                   min(p0[i], p1[i]) < np.max(obj[:, i])
                   for i in range(len(p0))]
            if all(tmp):
                valid_objects.append(obj)

        # Frame collision pre-check
        tmp = [max(p0[i], p1[i]) > np.min(self.frame[:, i]) and
               min(p0[i], p1[i]) < np.max(self.frame[:, i])
               for i in range(len(p0))]
        if all(tmp):
            valid_objects.append(self.frame)

        for obj in valid_objects:
            for i in range(len(obj) - 1):
                is_valid, r, s = self.calc_intersection_point(
                    obj[i], obj[i + 1], p0, p1)
                if not is_valid:
                    return False
        return True

    def line_validation_for_grid(self, p0, p1):
        # Set searching point
        p_vector = p1 - p0
        dist = np.linalg.norm(p_vector)
        if dist > 1e-8:
            imax = int(dist / self.delta) + 1
            pts = [p0 + p_vector / dist * self.delta * i for i in range(imax)]
            pts.append(p1)
            np.random.shuffle(np.array(pts))
        else:
            pts = [p0]

        # Check collision
        for p in pts:
            res = self.point_validation(p)
            if not res:
                break
        return res

    def path_validation(self, nodelist):
        for q0, q1 in zip(nodelist[:-1], nodelist[1:]):
            if not self.line_validation(q0, q1):
                return False
        return True

    def calc_intersection_point(self, A, B, C, D):
        denominator = ((B[0] - A[0]) * (C[1] - D[1])
                       - (B[1] - A[1]) * (C[0] - D[0]))
        # If two lines are parallel,
        if abs(denominator) < 1e-6:
            return True, None, None
        AC = A - C
        r = ((D[1] - C[1]) * AC[0] - (D[0] - C[0]) * AC[1]) / denominator
        s = ((B[1] - A[1]) * AC[0] - (B[0] - A[0]) * AC[1]) / denominator
        # If the intersection is out of the edges
        if r < -1e-6 or r > 1.00001 or s < -1e-6 or s > 1.00001:
            return True, r, s
        # If the intersection exists,
        return False, r, s

    def move_points(self, p):

        def pt_line_dist(a, b, c):
            u = b - a
            v = c - a
            L = np.cross(u, v) / np.linalg.norm(u)
            w = np.array([u[1], -u[0]])
            w = w / np.linalg.norm(w)
            d = c + 2.0 * L * w
            return abs(L), d

        # If the point is in the objects, it moves outside.
        target_obj = None
        for obj in self.objects:
            if not self.crossing_number_algorithm(p, obj):
                target_obj = obj

        if target_obj is None:
            return p
        else:
            minimum = np.inf
            for i in range(len(target_obj) - 1):
                dist, pt = pt_line_dist(target_obj[i], target_obj[i + 1], p)
                if dist < minimum:
                    new_pt = pt
            return new_pt

    def add_points(self, p0, p1):

        # Narrow down valid objects
        valid_objects = []
        for obj in self.objects:
            tmp = [max(p0[i], p1[i]) > np.min(obj[:, i]) and
                   min(p0[i], p1[i]) < np.max(obj[:, i])
                   for i in range(len(p0))]
            if all(tmp):
                valid_objects.append(obj)

        # Assume that both edges are out of the objects
        insert_pts = []
        s_list = []
        for obj in valid_objects:
            inter_section = []
            vertices = []
            for i in range(len(obj) - 1):
                is_valid, r, s = self.calc_intersection_point(
                    obj[i], obj[i + 1], p0, p1)
                if not is_valid:
                    vertices.append(i)
                    inter_section.append(s)

            # Object is not convex.
            if len(vertices) > 2:
                print('[Warning: Collision Avoidance Failure] \
                      Object is not convex.')
                insert_pts.append(None)
                # s_list.append(None)

            # Object is convex.
            elif len(vertices) == 2:
                # Path intersects with two adjacent lines
                if vertices[1] - vertices[0] == 1:
                    insert_pts.append(obj[vertices[1]])
                    s_list.append(inter_section[0])
                elif vertices[1] - vertices[0] == len(obj) - 2:
                    insert_pts.append(obj[vertices[0]])
                    s_list.append(inter_section[0])
                # Path intersects with two non-adjacent lines
                else:
                    print(
                        '[Warning: Collision Avoidance Failure] \
                         Non-adjacent lines')
                    insert_pts.append(None)
                    # s_list.append(None)
        insert_pts = np.array(insert_pts)
        insert_pts = insert_pts[np.argsort(s_list)]
        return insert_pts

    def add_points2(self, pts):
        iter = 0
        path_is_valid = True

        while iter + 1 < len(pts):
            p0 = pts[iter]
            p1 = pts[iter + 1]
            # Narrow down valid objects
            valid_objects = []
            for obj in self.objects:
                tmp = [max(p0[i], p1[i]) > np.min(obj[:, i]) and
                       min(p0[i], p1[i]) < np.max(obj[:, i])
                       for i in range(len(p0))]
                if all(tmp):
                    valid_objects.append(obj)

            # Assume that both edges are out of the objects
            insert_pts = []
            s_list = []
            for obj in valid_objects:
                inter_section = []
                vertices = []
                for i in range(len(obj) - 1):
                    is_valid, r, s = self.calc_intersection_point(
                        obj[i], obj[i + 1], p0, p1)
                    if not is_valid:
                        vertices.append(i)
                        inter_section.append(s)

                # Object is not convex.
                if len(vertices) > 2:
                    print('[Warning: Collision Avoidance Failure] \
                          Object is not convex.')
                    path_is_valid = False

                # Object is convex.
                elif len(vertices) == 2:
                    # Path intersects with two adjacent lines
                    if vertices[1] - vertices[0] == 1:
                        insert_pts.append(obj[vertices[1]])
                        s_list.append(inter_section[0])
                    elif vertices[1] - vertices[0] == len(obj) - 2:
                        insert_pts.append(obj[vertices[0]])
                        s_list.append(inter_section[0])
                    # Path intersects with two non-adjacent lines
                    else:
                        print(
                            '[Warning: Collision Avoidance Failure] \
                             Non-adjacent lines')
                        path_is_valid = False

            insert_pts = np.array(insert_pts)
            s_list = np.array(s_list)
            insert_pts = insert_pts[np.argsort(s_list)]

            if path_is_valid:
                for j, pt in enumerate(insert_pts):
                    pts = np.insert(pts, j + iter + 1, pt, axis=0)
                iter += 1
            else:
                pts = None
                break
        return pts

    def collision_avoidance(self, path):
        # Move points in the objects.
        tmp = np.array([self.move_points(p) for p in path])
        tmp = self.add_points2(tmp)

        # Add points so that lines do not intersect with the objects.
        '''
        i = 0
        while i + 1 < len(tmp):
            insert_pts = self.add_points(tmp[i], tmp[i + 1])
            if None not in insert_pts:
                for j, pt in enumerate(insert_pts):
                    tmp = np.insert(tmp, j + i + 1, pt, axis=0)
                i += 1
            else:
                tmp = None
                break
        '''
        return tmp
