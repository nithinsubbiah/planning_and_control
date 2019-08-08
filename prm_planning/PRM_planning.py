import os
import sys
import time
from collections import defaultdict

sys.path.append(os.getcwd())
from utilities import vrep_utils as vu
import locobot_joint_ctrl
import forward_kinematics as fk
import CollisionDetection
import transformations as tf
import numpy as np


try:
    from lib import vrep
except:
    print ('"vrep.py" could not be imported. Check for the library file')


def BFS(graph, vertices, start, goal):

    visited = [False] * (len(vertices))
    queue = []
    queue.append(start.tolist())
    path = []


    while len(queue)!=0 and visited[-1]==False:

        if path == []:
            path = [queue.pop(0)]
        else:
            path = queue.pop(0)
        element = path[-1]
        #if(np.array_equal(element,goal)):
        if element==goal.tolist():
            return path

        element_pos = np.argwhere([np.all((element-vertices)==0,axis=1)])[0][1]
        visited[element_pos] = True

        for i in graph[element_pos]:
            if visited[i] == False:
                new_path = path[:]
                new_path.append(vertices[i].tolist())
                queue.append(new_path)

    print("No path found, Run the program again to find a path\n")

    import pdb; pdb.set_trace()

    sys.exit()



def transformation_with_only_offset(offset):              #create a transformation matrix with no rotation but only translation
    T = np.zeros((4,4))
    T[0:3,0:3] = np.eye(3)
    T[3] = [0,0,0,1]
    T[0:3,3] = offset

    return T

def vertice_collision_checker(joint_targets,obstacle,joint_to_center,dimensions):                        #function that takes in configurations and checks if its in collision

    T1,T2,T3,T4,T5 = fk.getWristPose(joint_targets[0:5]) #using FK from assignment 1
    #finding the origins and orientations of the joint cuboids at joint targets
    T_fk = []
    T_fk.extend((T1,T2,T3,T4,T5))
    origins = []
    orientations = []

    #this loop finds origin and orientation for 5 joint cuboids
    for i in range(5):

        T_joint_to_center = transformation_with_only_offset(joint_to_center[i])
        T = np.matmul(T_fk[i],T_joint_to_center)

        x,y,z = tf.euler_from_matrix(T[0:3,0:3])
        angles = np.array((x,y,z))
        origins.append(T[0:3,3])
        orientations.append(angles)

    #to find origin and orientation for joint 6&7
    joint_5_to_finger = np.array(((0.0728,joint_targets[5],0.0051),(0.0728,joint_targets[6],0.0051)))

    for i in range(5,7):
        #need to find T_joint_to_finger. Have translations, what should be the rotations? eye(3)?
        T_joint_5_to_finger = transformation_with_only_offset(joint_5_to_finger[i-5])        #check if right
        T_joint_to_center = transformation_with_only_offset(joint_to_center[i])
        T = np.matmul(T_joint_5_to_finger,T_joint_to_center)
        x,y,z = tf.euler_from_matrix(T[0:3,0:3])
        angles = np.array((x,y,z))
        origins.append(T[0:3,3])
        orientations.append(angles)

    origins = np.array(origins)
    orientations = np.array(orientations)   #origin, orientation and dimensions for the link cuboids are known for the joint configuration

    #Instantiate cuboid object for link cuboids from CollisionDetection
    link_cuboid = []
    for i in range(7):
        link_cuboid.append(CollisionDetection.Cuboid(origins[i],orientations[i],dimensions[i]))

    #Check for collision
    for i in range(7):
        for j in range(6):
            collision = link_cuboid[i].check_collision(obstacle[j])
            if collision == True:
                break
        if collision == True:
            break

    return collision


def main():

    dimensions = np.load('dimensions.npy')
    joint_to_center = np.load('joint_to_center_offset.npy')

    #obstacle_dimensions = np.load('obstacle_dimensions.npy')
    #obstacle_orientations = np.load('obstacle_orientations.npy')
    #obstacle_origins = np.load('obstacle_origins.npy')

    obstacle_dimensions = np.array((0.072,0.098,0.198))
    obstacle_orientations = np.array((0,0,0))
    obstacle_origins = np.array((0.2,0.2,0.09))



    #Instantiate cuboid object for obstacle cuboids from CollisionDetection
    obstacle = []
    for i in range(6):
        obstacle.append(CollisionDetection.Cuboid(obstacle_origins[i],obstacle_orientations[i],obstacle_dimensions[i]))


    no_vertices = 50                                                            #No. of vertices to add to graph during training phase
    n = 0
    vertices = []
    deg_to_rad = np.pi/180.

    while(n<no_vertices):
        pi = np.pi

        joint_targets = [np.random.uniform(low=-pi/2, high=pi/2),               #random sampling joint angles
                         np.random.uniform(low=-pi/2, high=pi/2),
                         np.random.uniform(low=-pi/2, high=pi/2),
                         np.random.uniform(low=-pi/2, high=pi/2),
                         np.random.uniform(low=-pi/2, high=pi/2),
                         -0.03,
                         0.03]

        collision = vertice_collision_checker(joint_targets,obstacle,joint_to_center,dimensions)
        #add vertices if no collision
        if collision == False:
            vertices.append(joint_targets)
            n+=1

    start = [-80*deg_to_rad,0,0,0,0,-0.03,0.03]
    goal = [0,60*deg_to_rad,-75*deg_to_rad,-75*deg_to_rad,0,-0.03,0.03]
    #start = [0,0,0,0,0,-0.03,0.03]
    #goal = [pi/2,0,0,0,0,-0.03,0.03]
    vertices.append(start)
    vertices.append(goal)

    vertices = np.array(vertices)

    #Find edges that are not in collision
    k = 5         #No. of Nearest Neighbours
    p = 10        #No. of points sampled between the edges
    edges = []

    for i in range(vertices.shape[0]):
        dist = np.linalg.norm(vertices[i]-vertices,axis=1)                      #euclidean distance between one and all vertices
        sorted_dist = np.argsort(dist)

        #To find the k-NN and if the path between is not in collision add the edge
        for j in range(k):
            idx = sorted_dist[j+1]
            edge_start = vertices[i]
            edge_end = vertices[idx]
            sample_points = np.linspace(edge_start,edge_end,num=p)

            for p in range(sample_points.shape[0]):                             #check if each path is in collision with obstacles
                collision = vertice_collision_checker(sample_points[p],obstacle,joint_to_center,dimensions)
                if collision == True:
                    break

            if collision == False:
                edges.extend((edge_start,edge_end))

    edges = np.array(edges)

    graph = defaultdict(list)

    start = np.array(start)
    goal = np.array(goal)

    for i in range(0,edges.shape[0],2):
        j = i+1
        x = np.argwhere([np.all((edges[i]-vertices)==0, axis=1)])[0][1]
        y = np.argwhere([np.all((edges[j]-vertices)==0, axis=1)])[0][1]
        graph[x].append(y)

    planner_positions = BFS(graph,vertices,start,goal)

    print("Path found:\n")
    print(planner_positions)

    print ('Connecting to V-REP...')
    clientID = vu.connect_to_vrep()
    print ('Connected.')

    # Reset simulation in case something was running
    vu.reset_sim(clientID)

    # Initial control inputs are zero
    vu.set_arm_joint_target_velocities(clientID, np.zeros(vu.N_ARM_JOINTS))

    # Despite the name, this sets the maximum allowable joint force
    vu.set_arm_joint_forces(clientID, 50.*np.ones(vu.N_ARM_JOINTS))

    # One step to process the above settings
    vu.step_sim(clientID)

    controller = locobot_joint_ctrl.ArmController()
    i=0

    for target in planner_positions:

        # Set new target position
        controller.set_target_joint_positions(target)

        steady_state_reached = False
        while not steady_state_reached:

            timestamp = vu.get_sim_time_seconds(clientID)
            #print('Simulation time: {} sec'.format(timestamp))

            # Get current joint positions
            sensed_joint_positions = vu.get_arm_joint_positions(clientID)

            # Calculate commands
            commands = controller.calculate_commands_from_feedback(timestamp, sensed_joint_positions)

            # Send commands to V-REP
            vu.set_arm_joint_target_velocities(clientID, commands)

            # Print current joint positions (comment out if you'd like)
            #print(sensed_joint_positions)
            vu.step_sim(clientID, 1)

            # Determine if we've met the condition to move on to the next point
            steady_state_reached = controller.has_stably_converged_to_target()

        i+=1
        print(i)

    vu.stop_sim(clientID)



if __name__=='__main__':
    main()
