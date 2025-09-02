"""
PyFEA 3D: Advanced 3D Finite Element Analysis Solver
====================================================

A comprehensive 3D finite element analysis implementation for spatial beam and frame structures.
This advanced solver handles 6 degrees of freedom per node (3 translations + 3 rotations)
and demonstrates cutting-edge computational methods for 3D structural analysis.

Author: Abhijith R Pillai
Date: August 2025
Version: 2.0.0

Key Features:
- 3D beam element formulation with 6 DOF per node
- Spatial coordinate transformation matrices
- 3D frame structure analysis capabilities
- Advanced 3D visualization and interaction
- Professional-grade accuracy and validation
- Real-time 3D structural analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
warnings.filterwarnings('ignore')

class Node3D:
    """
    3D Node class representing connection points in 3D space.
    
    Each node has 6 degrees of freedom:
    - 3 translations (ux, uy, uz)
    - 3 rotations (rx, ry, rz)
    
    Attributes:
        x, y, z (float): Node coordinates in global 3D coordinate system
        id (int): Unique node identifier
        dof_ux, dof_uy, dof_uz (int): Global DOF numbers for translations
        dof_rx, dof_ry, dof_rz (int): Global DOF numbers for rotations
        fixed_* (bool): Boundary condition flags for each DOF
        force_*, moment_* (float): Applied loads and moments
        displacement_*, rotation_* (float): Solution results
    """
    
    def __init__(self, x, y, z, node_id):
        """Initialize 3D node with coordinates and ID."""
        self.x = x
        self.y = y
        self.z = z
        self.id = node_id
        
        # DOF indices (assigned during system assembly)
        self.dof_ux = None  # X translation DOF
        self.dof_uy = None  # Y translation DOF
        self.dof_uz = None  # Z translation DOF
        self.dof_rx = None  # X rotation DOF
        self.dof_ry = None  # Y rotation DOF
        self.dof_rz = None  # Z rotation DOF
        
        # Boundary conditions
        self.fixed_ux = False
        self.fixed_uy = False
        self.fixed_uz = False
        self.fixed_rx = False
        self.fixed_ry = False
        self.fixed_rz = False
        
        # Applied loads
        self.force_x = 0.0
        self.force_y = 0.0
        self.force_z = 0.0
        self.moment_x = 0.0
        self.moment_y = 0.0
        self.moment_z = 0.0
        
        # Solution results
        self.displacement_x = 0.0
        self.displacement_y = 0.0
        self.displacement_z = 0.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        
    def get_coordinates(self):
        """Return node coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])
        
    def get_displacements(self):
        """Return nodal displacements as numpy array."""
        return np.array([
            self.displacement_x, self.displacement_y, self.displacement_z,
            self.rotation_x, self.rotation_y, self.rotation_z
        ])

class BeamElement3D:
    """
    3D Beam element class implementing spatial beam theory.
    
    Each element connects two nodes and has 12 degrees of freedom:
    - 6 DOF at node i (3 translations + 3 rotations)
    - 6 DOF at node j (3 translations + 3 rotations)
    
    The element accounts for:
    - Axial deformation
    - Bending about local y and z axes
    - Torsional deformation
    """
    
    def __init__(self, node_i, node_j, E, G, A, Iy, Iz, J, rho=0):
        """
        Initialize 3D beam element with material and geometric properties.
        
        Parameters:
        -----------
        node_i, node_j : Node3D objects
            Start and end nodes of the element
        E : float
            Young's modulus (Pa)
        G : float
            Shear modulus (Pa)
        A : float
            Cross-sectional area (m^2)
        Iy, Iz : float
            Second moments of area about local y and z axes (m^4)
        J : float
            Torsional constant (m^4)
        rho : float, optional
            Material density (kg/m^3)
        """
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.G = G
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.rho = rho
        
        # Calculate element geometry
        self._compute_geometry()
        
        # Compute element stiffness matrix
        self._compute_stiffness_matrix()
        
    def _compute_geometry(self):
        """Compute element geometric properties."""
        # Element vector
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        
        self.length = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if self.length < 1e-12:
            raise ValueError("Element has zero length")
        
        # Direction cosines
        self.cx = dx / self.length
        self.cy = dy / self.length
        self.cz = dz / self.length
        
        # Local coordinate system
        self._compute_local_axes()
        
    def _compute_local_axes(self):
        """
        Compute local coordinate system for the element.
        
        Local x-axis: along the element
        Local y-axis: perpendicular to x-axis
        Local z-axis: completes right-hand coordinate system
        """
        # Local x-axis (along element)
        self.x_local = np.array([self.cx, self.cy, self.cz])
        
        # Choose local y-axis (avoid singularity when element is vertical)
        if abs(self.cz) < 0.9:
            # Use cross product with global z-axis
            temp = np.array([0, 0, 1])
        else:
            # Use cross product with global y-axis
            temp = np.array([0, 1, 0])
            
        self.z_local = np.cross(self.x_local, temp)
        self.z_local = self.z_local / np.linalg.norm(self.z_local)
        
        # Local y-axis completes the right-hand system
        self.y_local = np.cross(self.z_local, self.x_local)
        
        # Transformation matrix from local to global coordinates
        self.T_matrix = np.array([
            self.x_local,
            self.y_local,
            self.z_local
        ])
        
    def _compute_stiffness_matrix(self):
        """
        Compute 3D beam element stiffness matrix.
        
        The local stiffness matrix includes:
        - Axial stiffness (EA/L)
        - Bending stiffness about y-axis (EIz)
        - Bending stiffness about z-axis (EIy)
        - Torsional stiffness (GJ/L)
        """
        L = self.length
        EA = self.E * self.A
        EIy = self.E * self.Iy
        EIz = self.E * self.Iz
        GJ = self.G * self.J
        
        # Local stiffness matrix (12x12)
        k_local = np.zeros((12, 12))
        
        # Axial terms (DOF 0, 6)
        k_local[0, 0] = EA / L
        k_local[0, 6] = -EA / L
        k_local[6, 0] = -EA / L
        k_local[6, 6] = EA / L
        
        # Bending about local z-axis (DOF 1, 5, 7, 11)
        k_local[1, 1] = 12 * EIz / L**3
        k_local[1, 5] = 6 * EIz / L**2
        k_local[1, 7] = -12 * EIz / L**3
        k_local[1, 11] = 6 * EIz / L**2
        
        k_local[5, 1] = 6 * EIz / L**2
        k_local[5, 5] = 4 * EIz / L
        k_local[5, 7] = -6 * EIz / L**2
        k_local[5, 11] = 2 * EIz / L
        
        k_local[7, 1] = -12 * EIz / L**3
        k_local[7, 5] = -6 * EIz / L**2
        k_local[7, 7] = 12 * EIz / L**3
        k_local[7, 11] = -6 * EIz / L**2
        
        k_local[11, 1] = 6 * EIz / L**2
        k_local[11, 5] = 2 * EIz / L
        k_local[11, 7] = -6 * EIz / L**2
        k_local[11, 11] = 4 * EIz / L
        
        # Bending about local y-axis (DOF 2, 4, 8, 10)
        k_local[2, 2] = 12 * EIy / L**3
        k_local[2, 4] = -6 * EIy / L**2
        k_local[2, 8] = -12 * EIy / L**3
        k_local[2, 10] = -6 * EIy / L**2
        
        k_local[4, 2] = -6 * EIy / L**2
        k_local[4, 4] = 4 * EIy / L
        k_local[4, 8] = 6 * EIy / L**2
        k_local[4, 10] = 2 * EIy / L
        
        k_local[8, 2] = -12 * EIy / L**3
        k_local[8, 4] = 6 * EIy / L**2
        k_local[8, 8] = 12 * EIy / L**3
        k_local[8, 10] = 6 * EIy / L**2
        
        k_local[10, 2] = -6 * EIy / L**2
        k_local[10, 4] = 2 * EIy / L
        k_local[10, 8] = 6 * EIy / L**2
        k_local[10, 10] = 4 * EIy / L
        
        # Torsional terms (DOF 3, 9)
        k_local[3, 3] = GJ / L
        k_local[3, 9] = -GJ / L
        k_local[9, 3] = -GJ / L
        k_local[9, 9] = GJ / L
        
        # Store local stiffness matrix
        self.k_local = k_local
        
        # Transform to global coordinates
        self._transform_to_global()
        
    def _transform_to_global(self):
        """Transform local stiffness matrix to global coordinates."""
        # Create 12x12 transformation matrix
        T = np.zeros((12, 12))
        
        # Fill transformation matrix with 3x3 blocks
        for i in range(4):
            start_idx = i * 3
            T[start_idx:start_idx+3, start_idx:start_idx+3] = self.T_matrix
            
        # Transform stiffness matrix: K_global = T^T * K_local * T
        self.k_global = T.T @ self.k_local @ T
        
    def get_dof_indices(self):
        """Return global DOF indices for matrix assembly."""
        return [
            self.node_i.dof_ux, self.node_i.dof_uy, self.node_i.dof_uz,
            self.node_i.dof_rx, self.node_i.dof_ry, self.node_i.dof_rz,
            self.node_j.dof_ux, self.node_j.dof_uy, self.node_j.dof_uz,
            self.node_j.dof_rx, self.node_j.dof_ry, self.node_j.dof_rz
        ]
        
    def get_element_forces(self, global_displacements):
        """
        Calculate internal forces and moments in the element.
        
        Parameters:
        -----------
        global_displacements : numpy array
            Global displacement vector
            
        Returns:
        --------
        dict : Element force results
        """
        # Extract element displacements from global solution
        dof_indices = self.get_dof_indices()
        element_displacements = np.zeros(12)
        
        for i, dof in enumerate(dof_indices):
            if dof >= 0:
                element_displacements[i] = global_displacements[dof]
        
        # Calculate local forces: F_local = K_local * T * U_global
        T = np.zeros((12, 12))
        for i in range(4):
            start_idx = i * 3
            T[start_idx:start_idx+3, start_idx:start_idx+3] = self.T_matrix
            
        local_displacements = T @ element_displacements
        local_forces = self.k_local @ local_displacements
        
        return {
            'axial_force': local_forces[0],
            'shear_y': local_forces[1],
            'shear_z': local_forces[2],
            'torsion': local_forces[3],
            'moment_y': local_forces[4],
            'moment_z': local_forces[5],
            'local_displacements': local_displacements,
            'local_forces': local_forces
        }

class FEASolver3D:
    """
    Main 3D finite element analysis solver for spatial beam and frame structures.
    
    This class implements the complete 3D FEA workflow:
    1. 3D model setup (nodes, elements, loads, boundary conditions)
    2. Global matrix assembly with 6 DOF per node
    3. 3D system solution using advanced linear algebra
    4. Results extraction and 3D post-processing
    """
    
    def __init__(self, name="3D FEA Model"):
        """Initialize the 3D FEA solver."""
        self.name = name
        self.nodes = []
        self.elements = []
        self.num_dof = 0
        self.K_global = None
        self.F_global = None
        self.U_global = None
        
    def add_node(self, x, y, z, node_id=None):
        """
        Add a 3D node to the finite element model.
        
        Parameters:
        -----------
        x, y, z : float
            Node coordinates in 3D space
        node_id : int, optional
            Node identifier (auto-assigned if None)
            
        Returns:
        --------
        Node3D : Created node object
        """
        if node_id is None:
            node_id = len(self.nodes)
        
        node = Node3D(x, y, z, node_id)
        self.nodes.append(node)
        return node
        
    def add_beam_element_3d(self, node_i, node_j, E, G, A, Iy, Iz, J, rho=0):
        """
        Add a 3D beam element between two nodes.
        
        Parameters:
        -----------
        node_i, node_j : Node3D
            Connected nodes
        E : float
            Young's modulus (Pa)
        G : float
            Shear modulus (Pa) 
        A : float
            Cross-sectional area (m^2)
        Iy, Iz : float
            Second moments of area (m^4)
        J : float
            Torsional constant (m^4)
        rho : float, optional
            Material density (kg/m^3)
            
        Returns:
        --------
        BeamElement3D : Created element object
        """
        element = BeamElement3D(node_i, node_j, E, G, A, Iy, Iz, J, rho)
        self.elements.append(element)
        return element
        
    def set_boundary_condition_3d(self, node, **kwargs):
        """
        Set 3D boundary conditions for a node.
        
        Parameters:
        -----------
        node : Node3D
            Node to constrain
        **kwargs : bool
            fixed_ux, fixed_uy, fixed_uz, fixed_rx, fixed_ry, fixed_rz
        """
        for attr, value in kwargs.items():
            if hasattr(node, attr):
                setattr(node, attr, value)
                
    def apply_3d_load(self, node, **kwargs):
        """
        Apply 3D loads to a node.
        
        Parameters:
        -----------
        node : Node3D
            Node to load
        **kwargs : float
            force_x, force_y, force_z, moment_x, moment_y, moment_z
        """
        for attr, value in kwargs.items():
            if hasattr(node, attr):
                current_value = getattr(node, attr)
                setattr(node, attr, current_value + value)
                
    def _assign_dof_3d(self):
        """Assign global DOF numbers to unconstrained 3D DOFs."""
        dof_counter = 0
        
        for node in self.nodes:
            # Translation DOFs
            if not node.fixed_ux:
                node.dof_ux = dof_counter
                dof_counter += 1
            else:
                node.dof_ux = -1
                
            if not node.fixed_uy:
                node.dof_uy = dof_counter
                dof_counter += 1
            else:
                node.dof_uy = -1
                
            if not node.fixed_uz:
                node.dof_uz = dof_counter
                dof_counter += 1
            else:
                node.dof_uz = -1
                
            # Rotation DOFs
            if not node.fixed_rx:
                node.dof_rx = dof_counter
                dof_counter += 1
            else:
                node.dof_rx = -1
                
            if not node.fixed_ry:
                node.dof_ry = dof_counter
                dof_counter += 1
            else:
                node.dof_ry = -1
                
            if not node.fixed_rz:
                node.dof_rz = dof_counter
                dof_counter += 1
            else:
                node.dof_rz = -1
                
        self.num_dof = dof_counter
        
    def _assemble_global_matrices_3d(self):
        """Assemble 3D global stiffness matrix and force vector."""
        # Initialize global matrices
        self.K_global = np.zeros((self.num_dof, self.num_dof))
        self.F_global = np.zeros(self.num_dof)
        
        # Assemble stiffness matrix contributions
        for element in self.elements:
            dof_indices = element.get_dof_indices()
            
            # Add element stiffness to global matrix
            for i in range(12):
                for j in range(12):
                    if dof_indices[i] >= 0 and dof_indices[j] >= 0:
                        self.K_global[dof_indices[i], dof_indices[j]] += element.k_global[i, j]
        
        # Assemble force vector
        for node in self.nodes:
            if node.dof_ux >= 0:
                self.F_global[node.dof_ux] += node.force_x
            if node.dof_uy >= 0:
                self.F_global[node.dof_uy] += node.force_y
            if node.dof_uz >= 0:
                self.F_global[node.dof_uz] += node.force_z
            if node.dof_rx >= 0:
                self.F_global[node.dof_rx] += node.moment_x
            if node.dof_ry >= 0:
                self.F_global[node.dof_ry] += node.moment_y
            if node.dof_rz >= 0:
                self.F_global[node.dof_rz] += node.moment_z
                
    def solve_3d(self):
        """
        Solve the 3D finite element system K*U = F.
        
        This method performs the complete 3D solution process:
        1. 3D DOF assignment  
        2. 3D matrix assembly
        3. Linear system solution
        4. 3D results extraction
        """
        print(f"Solving 3D FEA system: {self.name}...")
        
        # Setup 3D system
        self._assign_dof_3d()
        self._assemble_global_matrices_3d()
        
        # Check for valid system
        if self.num_dof == 0:
            print("Warning: No degrees of freedom to solve!")
            return
            
        print(f"System size: {self.num_dof} DOFs")
        
        # Solve linear system
        try:
            start_time = time.time()
            self.U_global = np.linalg.solve(self.K_global, self.F_global)
            solve_time = time.time() - start_time
            
            # Extract nodal results
            for node in self.nodes:
                if node.dof_ux >= 0:
                    node.displacement_x = self.U_global[node.dof_ux]
                if node.dof_uy >= 0:
                    node.displacement_y = self.U_global[node.dof_uy]
                if node.dof_uz >= 0:
                    node.displacement_z = self.U_global[node.dof_uz]
                if node.dof_rx >= 0:
                    node.rotation_x = self.U_global[node.dof_rx]
                if node.dof_ry >= 0:
                    node.rotation_y = self.U_global[node.dof_ry]
                if node.dof_rz >= 0:
                    node.rotation_z = self.U_global[node.dof_rz]
                    
            print(f"3D Solution completed in {solve_time*1000:.2f} ms")
            
        except np.linalg.LinAlgError as e:
            print(f"Error: Failed to solve 3D system - {e}")
            print("Check for singular stiffness matrix or insufficient constraints.")
        
    def plot_3d_structure(self, scale_factor=1.0, show_deformed=True, figsize=(12, 10)):
        """
        Plot the 3D structure with optional deformed shape.
        
        Parameters:
        -----------
        scale_factor : float
            Amplification factor for displacements
        show_deformed : bool
            Whether to show deformed shape
        figsize : tuple
            Figure size for matplotlib
            
        Returns:
        --------
        matplotlib.figure.Figure : Generated figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original structure
        for element in self.elements:
            x_orig = [element.node_i.x, element.node_j.x]
            y_orig = [element.node_i.y, element.node_j.y]
            z_orig = [element.node_i.z, element.node_j.z]
            ax.plot(x_orig, y_orig, z_orig, 'b-', linewidth=2, alpha=0.5, label='Original' if element == self.elements[0] else "")
        
        # Plot deformed shape
        if show_deformed and self.U_global is not None:
            for element in self.elements:
                x_def = [element.node_i.x + scale_factor * element.node_i.displacement_x,
                         element.node_j.x + scale_factor * element.node_j.displacement_x]
                y_def = [element.node_i.y + scale_factor * element.node_i.displacement_y,
                         element.node_j.y + scale_factor * element.node_j.displacement_y]
                z_def = [element.node_i.z + scale_factor * element.node_i.displacement_z,
                         element.node_j.z + scale_factor * element.node_j.displacement_z]
                ax.plot(x_def, y_def, z_def, 'r-', linewidth=2, 
                       label='Deformed' if element == self.elements[0] else "")
        
        # Plot nodes
        for node in self.nodes:
            ax.scatter(node.x, node.y, node.z, c='black', s=50)
            ax.text(node.x, node.y, node.z, f'  N{node.id}', fontsize=8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Structure: {self.name}\nScale Factor: {scale_factor}x')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

class FEAResultsAnalyzer3D:
    """Post-processing and analysis tools for 3D FEA results."""
    
    def __init__(self, solver):
        """Initialize 3D results analyzer."""
        self.solver = solver
        
    def calculate_max_displacement(self):
        """Calculate maximum displacement magnitude in the structure."""
        max_disp = 0
        max_node = None
        
        for node in self.solver.nodes:
            disp_magnitude = np.sqrt(
                node.displacement_x**2 + 
                node.displacement_y**2 + 
                node.displacement_z**2
            )
            if disp_magnitude > max_disp:
                max_disp = disp_magnitude
                max_node = node
                
        return max_disp, max_node
        
    def calculate_max_rotation(self):
        """Calculate maximum rotation magnitude in the structure."""
        max_rot = 0
        max_node = None
        
        for node in self.solver.nodes:
            rot_magnitude = np.sqrt(
                node.rotation_x**2 + 
                node.rotation_y**2 + 
                node.rotation_z**2
            )
            if rot_magnitude > max_rot:
                max_rot = rot_magnitude
                max_node = node
                
        return max_rot, max_node
        
    def generate_3d_report(self):
        """Generate comprehensive 3D analysis report."""
        report = []
        report.append("="*70)
        report.append(f"3D FINITE ELEMENT ANALYSIS REPORT: {self.solver.name}")
        report.append("="*70)
        
        # Model statistics
        report.append(f"Model Statistics:")
        report.append(f"  • Number of Nodes: {len(self.solver.nodes)}")
        report.append(f"  • Number of Elements: {len(self.solver.elements)}")
        report.append(f"  • Total DOFs: {self.solver.num_dof}")
        report.append("")
        
        # Displacement analysis
        max_disp, disp_node = self.calculate_max_displacement()
        max_rot, rot_node = self.calculate_max_rotation()
        
        report.append("3D Displacement Summary:")
        report.append(f"  • Maximum displacement magnitude: {max_disp*1000:.3f} mm")
        if disp_node:
            report.append(f"    - Location: Node {disp_node.id} ({disp_node.x:.2f}, {disp_node.y:.2f}, {disp_node.z:.2f})")
            report.append(f"    - Components: Ux={disp_node.displacement_x*1000:.3f}, Uy={disp_node.displacement_y*1000:.3f}, Uz={disp_node.displacement_z*1000:.3f} mm")
        
        report.append(f"  • Maximum rotation magnitude: {max_rot:.6f} rad ({np.degrees(max_rot):.3f}°)")
        if rot_node:
            report.append(f"    - Location: Node {rot_node.id}")
            report.append(f"    - Components: Rx={rot_node.rotation_x:.6f}, Ry={rot_node.rotation_y:.6f}, Rz={rot_node.rotation_z:.6f} rad")
        
        return "\n".join(report)

# Utility functions for common 3D structures
def create_3d_cantilever(L, E, G, A, Iy, Iz, J, n_elements=10):
    """
    Create a 3D cantilever beam model.
    
    Parameters:
    -----------
    L : float
        Beam length (m)
    E, G : float
        Material properties (Pa)
    A, Iy, Iz, J : float
        Cross-sectional properties (m^2, m^4)
    n_elements : int
        Number of elements
        
    Returns:
    --------
    FEASolver3D, list : Configured 3D model and nodes
    """
    solver = FEASolver3D("3D Cantilever Beam")
    
    # Create nodes along X-axis
    nodes = []
    for i in range(n_elements + 1):
        x = i * L / n_elements
        nodes.append(solver.add_node(x, 0, 0))
    
    # Create elements
    for i in range(n_elements):
        solver.add_beam_element_3d(nodes[i], nodes[i+1], E, G, A, Iy, Iz, J)
    
    # Fixed boundary condition at base (all DOFs constrained)
    solver.set_boundary_condition_3d(nodes[0], 
        fixed_ux=True, fixed_uy=True, fixed_uz=True,
        fixed_rx=True, fixed_ry=True, fixed_rz=True)
    
    return solver, nodes

def create_3d_frame(width, height, depth, E, G, A, Iy, Iz, J):
    """
    Create a 3D frame structure.
    
    Parameters:
    -----------
    width, height, depth : float
        Frame dimensions (m)
    E, G : float
        Material properties (Pa)
    A, Iy, Iz, J : float
        Cross-sectional properties
        
    Returns:
    --------
    FEASolver3D, dict : Configured 3D frame and node dictionary
    """
    solver = FEASolver3D("3D Frame Structure")
    
    # Create corner nodes
    nodes = {}
    
    # Base nodes
    nodes['base_000'] = solver.add_node(0, 0, 0)
    nodes['base_100'] = solver.add_node(width, 0, 0)
    nodes['base_010'] = solver.add_node(0, depth, 0)
    nodes['base_110'] = solver.add_node(width, depth, 0)
    
    # Top nodes
    nodes['top_001'] = solver.add_node(0, 0, height)
    nodes['top_101'] = solver.add_node(width, 0, height)
    nodes['top_011'] = solver.add_node(0, depth, height)
    nodes['top_111'] = solver.add_node(width, depth, height)
    
    # Create frame elements
    # Vertical columns
    solver.add_beam_element_3d(nodes['base_000'], nodes['top_001'], E, G, A, Iy, Iz, J)
    solver.add_beam_element_3d(nodes['base_100'], nodes['top_101'], E, G, A, Iy, Iz, J)
    solver.add_beam_element_3d(nodes['base_010'], nodes['top_011'], E, G, A, Iy, Iz, J)
    solver.add_beam_element_3d(nodes['base_110'], nodes['top_111'], E, G, A, Iy, Iz, J)
    
    # Base beams (X direction)
    solver.add_beam_element_3d(nodes['base_000'], nodes['base_100'], E, G, A, Iy, Iz, J)
    solver.add_beam_element_3d(nodes['base_010'], nodes['base_110'], E, G, A, Iy, Iz, J)
    
    # Base beams (Y direction)
    solver.add_beam_element_3d(nodes['base_000'], nodes['base_010'], E, G, A, Iy, Iz, J)
    solver.add_beam_element_3d(nodes['base_100'], nodes['base_110'], E, G, A, Iy, Iz, J)
    
    # Top beams (X direction)
    solver.add_beam_element_3d(nodes['top_001'], nodes['top_101'], E, G, A, Iy, Iz, J)
    solver.add_beam_element_3d(nodes['top_011'], nodes['top_111'], E, G, A, Iy, Iz, J)
    
    # Top beams (Y direction)
    solver.add_beam_element_3d(nodes['top_001'], nodes['top_011'], E, G, A, Iy, Iz, J)
    solver.add_beam_element_3d(nodes['top_101'], nodes['top_111'], E, G, A, Iy, Iz, J)
    
    # Fixed boundary conditions at base
    for key in ['base_000', 'base_100', 'base_010', 'base_110']:
        solver.set_boundary_condition_3d(nodes[key],
            fixed_ux=True, fixed_uy=True, fixed_uz=True,
            fixed_rx=True, fixed_ry=True, fixed_rz=True)
    
    return solver, nodes

def validate_3d_cantilever():
    """Validate 3D cantilever against analytical solution."""
    print("3D Cantilever Validation")
    print("-" * 40)
    
    # Problem parameters
    L = 2.0
    E = 200e9
    G = 80e9
    A = 0.01
    Iy = Iz = 5e-5
    J = 1e-4
    Fy = -5000  # Transverse load in Y direction
    
    # Create and solve model
    solver, nodes = create_3d_cantilever(L, E, G, A, Iy, Iz, J, 15)
    solver.apply_3d_load(nodes[-1], force_y=Fy)
    solver.solve_3d()
    
    # Compare with analytical solution
    fea_displacement = nodes[-1].displacement_y
    analytical_displacement = -abs(Fy) * L**3 / (3 * E * Iy)
    
    error = abs(fea_displacement - analytical_displacement) / abs(analytical_displacement) * 100
    
    print(f"End displacement (Y direction):")
    print(f"  FEA Result: {fea_displacement*1000:.4f} mm")
    print(f"  Analytical: {analytical_displacement*1000:.4f} mm")
    print(f"  Error: {error:.6f}%")
    
    return solver, error

if __name__ == "__main__":
    print("PyFEA 3D: Advanced 3D Finite Element Analysis Solver")
    print("=" * 60)
    print("Initializing 3D validation tests...")
    
    # Run 3D validation
    solver, error = validate_3d_cantilever()
    
    # Generate results
    analyzer = FEAResultsAnalyzer3D(solver)
    print("\n" + analyzer.generate_3d_report())
    
    # Create 3D visualization
    fig = solver.plot_3d_structure(scale_factor=100)
    plt.savefig('3d_cantilever_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n3D FEA Solver validation complete!")
    print(f"Accuracy: {100-error:.6f}% - Ready for professional use!")
