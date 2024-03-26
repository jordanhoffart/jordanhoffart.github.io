import meshpy.triangle as tr

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

import matplotlib.pyplot as plt
import pandas as pd

from typing import Callable


class ProblemData:
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        g: Callable[[np.ndarray], float],
        q: float,
    ) -> None:
        self.f: Callable[[np.ndarray], float] = f
        self.g: Callable[[np.ndarray], float] = g
        self.q: float = q


class Domain:
    def __init__(
        self,
        vertices: list[tuple[float, float]],
        edges: list[list[int]],
        holes: list[tuple[float, float]] | None = None,
    ) -> None:
        self.vertices: list[tuple[float, float]] = vertices
        self.edges: list[list[int]] = edges
        if holes is not None:
            self.holes: list[tuple[float, float]] = holes
        else:
            self.holes = []


def make_square() -> Domain:
    vertices: list[tuple[float, float]] = [(0, 0), (1, 0), (1, 1), (0, 1)]
    edges: list[list[int]] = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return Domain(vertices, edges)


def make_polygon() -> Domain:
    vertices: list[tuple[float, float]] = [(0, 0), (0.5, 0), (1, 1), (0, 2)]
    edges: list[list[int]] = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return Domain(vertices, edges)


def make_square_with_hole() -> Domain:
    vertices: list[tuple[float, float]] = [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.75, 0.75),
        (0.25, 0.75),
    ]
    edges: list[list[int]] = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
    ]
    holes: list[tuple[float, float]] = [(0.5, 0.5)]
    return Domain(vertices, edges, holes)


def compute_max_area(max_mesh_size: float, min_angle_deg: float = 20) -> float:
    return np.tan(min_angle_deg * np.pi / 180) / 4 * max_mesh_size**2


def triangulate_domain(
    domain: Domain, max_mesh_size: float, min_angle_deg: float = 20
) -> tr.MeshInfo:
    info = tr.MeshInfo()
    info.set_points(domain.vertices)
    info.set_facets(domain.edges)
    info.set_holes(domain.holes)
    max_area = compute_max_area(max_mesh_size, min_angle_deg)
    return tr.build(info, max_volume=max_area, min_angle=min_angle_deg)


def refine_mesh(
    mesh: tr.MeshInfo, max_mesh_size: float, min_angle_deg: float = 20
) -> tr.MeshInfo:
    mesh.element_volumes.setup()
    max_area = compute_max_area(max_mesh_size, min_angle_deg)
    for i in range(len(mesh.elements)):
        mesh.element_volumes[i] = max_area
    return tr.refine(mesh, min_angle=min_angle_deg)


def get_local_vertices(mesh: tr.MeshInfo, local_indices: list[int]) -> list[np.ndarray]:
    return [np.array(mesh.points[i]) for i in local_indices]


def compute_mesh_size(mesh: tr.MeshInfo) -> float:
    mesh_size: float = 0
    for cell in mesh.elements:
        cell_vertices: list[np.ndarray] = get_local_vertices(mesh, cell)
        cell_vertices.append(cell_vertices[0])
        for i in range(len(cell_vertices) - 1):
            edge_length: float = float(
                np.linalg.norm(cell_vertices[i + 1] - cell_vertices[i])
            )
            mesh_size = max(mesh_size, edge_length)
    return mesh_size


class Quadrature2D:
    def __init__(self) -> None:
        self.points: list[np.ndarray] = [
            np.array([0.66666666666666666667, 0.16666666666666666667]),
            np.array([0.16666666666666666667, 0.66666666666666666667]),
            np.array([0.16666666666666666667, 0.16666666666666666667]),
        ]  # FIXME
        self.weights: list[float] = [1 / 6, 1 / 6, 1 / 6]  # FIXME
        self.size: int = len(self.points)

    def point(self, q: int) -> np.ndarray:
        return self.points[q]

    def weight(self, q: int) -> float:
        return self.weights[q]


class Basis2D:
    def __init__(self) -> None:
        self.n_basis_functions: int = 3

    def value(self, i: int, x: np.ndarray) -> float:
        return (i == 0) * (1 - x[0] - x[1]) + (i == 1) * x[0] + (i == 2) * x[1]

    def gradient(self, i: int) -> np.ndarray:
        return (
            (i == 0) * np.array([-1, -1])
            + (i == 1) * np.array([1, 0])
            + (i == 2) * np.array([0, 1])
        )


class Mapping2D:
    def __init__(self, mesh: tr.MeshInfo) -> None:
        self.mesh: tr.MeshInfo = mesh
        self.v0: np.ndarray = np.zeros(2)
        self.J: np.ndarray = np.zeros((2, 2))
        self.detJ: float = 0
        self.invJ: np.ndarray = np.zeros((2, 2))

    def reinit(self, cell: list[int]) -> None:
        v0: np.ndarray
        v1: np.ndarray
        v2: np.ndarray
        v0, v1, v2 = get_local_vertices(self.mesh, cell)

        self.v0: np.ndarray = v0
        self.J: np.ndarray = np.array([v1 - v0, v2 - v0]).transpose()
        self.invJ: np.ndarray = np.linalg.inv(self.J)
        self.detJ: float = abs(np.linalg.det(self.J))

    def map(self, x: np.ndarray) -> np.ndarray:
        return self.v0 + np.dot(self.J, x)

    def transform(self, gradient: np.ndarray) -> np.ndarray:
        return np.matmul(gradient, self.invJ)

    def JxW(self, weight: float) -> float:
        return self.detJ * weight


class Quadrature1D:
    def __init__(self) -> None:
        self.points: list[float] = [
            0.5 + 0.5 * point for point in [np.sqrt(1 / 3), -np.sqrt(1 / 3)]
        ]  # FIXME
        self.weights: list[float] = [0.5, 0.5]  # FIXME
        self.size: int = len(self.points)

    def point(self, q: int) -> float:
        return self.points[q]

    def weight(self, q: int) -> float:
        return self.weights[q]


class Basis1D:
    def __init__(self) -> None:
        self.n_basis_functions: int = 2

    def value(self, i: int, x: float) -> float:
        return (i == 0) * (1 - x) + (i == 1) * x  # FIXME

    def derivative(self, i: int) -> float:
        return (i == 1) - (i == 0)  # FIXME


class Mapping1D2D:
    def __init__(self, mesh: tr.MeshInfo) -> None:
        self.mesh: tr.MeshInfo = mesh
        self.v0: np.ndarray = np.zeros(2)
        self.J: np.ndarray = np.zeros(2)
        self.lenJ: float = 0

    def reinit(self, edge: list[int]) -> None:
        v0: np.ndarray
        v1: np.ndarray
        v0, v1 = get_local_vertices(self.mesh, edge)

        self.v0: np.ndarray = v0  # FIXME
        self.J: np.ndarray = v1 - v0  # FIXME
        self.lenJ: float = float(np.linalg.norm(self.J))  # FIXME

    def map(self, x: float) -> np.ndarray:
        return self.v0 + x * self.J

    def JxW(self, weight: float) -> float:
        return self.lenJ * weight


class System:
    def __init__(self, n_dofs: int):
        self.n_dofs: int = n_dofs
        self.matrix: ss.lil_array = ss.lil_array((n_dofs, n_dofs))
        self.rhs: np.ndarray = np.zeros(n_dofs)


def assemble_cells(
    mesh: tr.MeshInfo, problem_data: ProblemData, system: System
) -> None:
    cell_quadrature: Quadrature2D = Quadrature2D()
    cell_basis: Basis2D = Basis2D()
    cell_mapping: Mapping2D = Mapping2D(mesh)

    for cell in mesh.elements:
        cell_mapping.reinit(cell)
        for p in range(cell_quadrature.size):
            q_p: np.ndarray = cell_quadrature.point(p)
            w_p: float = cell_quadrature.weight(p)
            f_p: float = problem_data.f(cell_mapping.map(q_p))
            JxW_p: float = cell_mapping.JxW(w_p)
            for i in range(cell_basis.n_basis_functions):
                row: int = cell[i]
                basis_ip: float = cell_basis.value(i, q_p)
                grad_basis_i: np.ndarray = cell_mapping.transform(
                    cell_basis.gradient(i)
                )
                system.rhs[row] += f_p * basis_ip * JxW_p
                for j in range(cell_basis.n_basis_functions):
                    col: int = cell[j]
                    basis_jp: float = cell_basis.value(j, q_p)  # FIXME
                    grad_basis_j: np.ndarray = cell_mapping.transform(
                        cell_basis.gradient(j)
                    )  # FIXME
                    system.matrix[row, col] += (
                        problem_data.q * basis_jp * basis_ip
                        + np.dot(grad_basis_j, grad_basis_i)
                    ) * JxW_p  # FIXME


def assemble_edges(
    mesh: tr.MeshInfo, problem_data: ProblemData, system: System
) -> None:
    edge_quadrature: Quadrature1D = Quadrature1D()
    edge_basis: Basis1D = Basis1D()
    edge_mapping: Mapping1D2D = Mapping1D2D(mesh)
    for edge in mesh.facets:
        edge_mapping.reinit(edge)
        for p in range(edge_quadrature.size):
            q_p: float = edge_quadrature.point(p)
            w_p: float = edge_quadrature.weight(p)
            JxW_p: float = edge_mapping.JxW(w_p)
            g_p: float = problem_data.g(edge_mapping.map(q_p))  # FIXME
            for i in range(edge_basis.n_basis_functions):
                row: int = edge[i]
                basis_ip: float = edge_basis.value(i, q_p)
                system.rhs[row] += g_p * basis_ip * JxW_p  # FIXME


def solve(
    mesh: tr.MeshInfo, problem_data: ProblemData, verbose: bool = True
) -> np.ndarray:
    n_dofs: int = len(mesh.points)
    if verbose:
        print("solving dofs =", n_dofs, "h = ", compute_mesh_size(mesh))
    system: System = System(n_dofs)
    assemble_cells(mesh, problem_data, system)
    assemble_edges(mesh, problem_data, system)
    if verbose:
        print("solved")
    return ssl.spsolve(system.matrix.tocsr(), system.rhs)


class ExactSolution:
    def __init__(
        self,
        exact_values: Callable[[np.ndarray], float],
        exact_gradients: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.value = exact_values
        self.gradient = exact_gradients
        pass


def compute_L2_error(
    mesh: tr.MeshInfo, solution: np.ndarray, exact_solution: ExactSolution
) -> float:
    L2_error_sq: float = 0
    cell_quadrature: Quadrature2D = Quadrature2D()
    cell_basis: Basis2D = Basis2D()
    cell_mapping: Mapping2D = Mapping2D(mesh)
    for cell in mesh.elements:
        cell_mapping.reinit(cell)
        for p in range(cell_quadrature.size):
            q_p: np.ndarray = cell_quadrature.point(p)
            w_p: float = cell_quadrature.weight(p)
            JxW_p: float = cell_mapping.JxW(w_p)
            exact_p: float = exact_solution.value(cell_mapping.map(q_p))
            difference = exact_p
            for i in range(cell_basis.n_basis_functions):
                row: int = cell[i]
                basis_ip: float = cell_basis.value(i, q_p)
                difference -= solution[row] * basis_ip
            L2_error_sq += JxW_p * difference**2
    return np.sqrt(L2_error_sq)


def compute_H1_error(
    mesh: tr.MeshInfo, solution: np.ndarray, exact_solution: ExactSolution
) -> float:
    H1_error_sq: float = compute_L2_error(mesh, solution, exact_solution) ** 2
    cell_quadrature: Quadrature2D = Quadrature2D()
    cell_basis: Basis2D = Basis2D()
    cell_mapping: Mapping2D = Mapping2D(mesh)
    for cell in mesh.elements:
        cell_mapping.reinit(cell)
        for p in range(cell_quadrature.size):
            q_p: np.ndarray = cell_quadrature.point(p)
            w_p: float = cell_quadrature.weight(p)
            JxW_p: float = cell_mapping.JxW(w_p)
            grad_exact_p: np.ndarray = exact_solution.gradient(cell_mapping.map(q_p))
            difference = grad_exact_p
            for i in range(cell_basis.n_basis_functions):
                row: int = cell[i]
                grad_basis_i: np.ndarray = cell_mapping.transform(
                    cell_basis.gradient(i)
                )
                difference -= solution[row] * grad_basis_i
            H1_error_sq += float(JxW_p * np.linalg.norm(difference) ** 2)
    return np.sqrt(H1_error_sq)


def plot_solution(
    mesh: tr.MeshInfo,
    solution: np.ndarray,
    show: bool = True,
    save: bool = False,
    filename: str | None = None,
) -> None:
    plt.figure()
    ax: plt.Axes = plt.axes(projection="3d")
    xs: list[float] = [v[0] for v in mesh.points]
    ys: list[float] = [v[1] for v in mesh.points]
    ax.plot_trisurf(xs, ys, solution, triangles=mesh.elements, cmap="viridis")
    if save:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.savefig("solution.png")
    if show:
        plt.show()
        plt.close()


class ErrorTable:
    def __init__(
        self, n_dofs: int, mesh_size: float, L2_error: float, H1_error: float
    ) -> None:
        self.dof_sizes: list[int] = [n_dofs]
        self.mesh_sizes: list[float] = [mesh_size]
        self.L2_errors: list[float] = [L2_error]
        self.H1_errors: list[float] = [H1_error]
        self.L2_rates: list[float] = [0]
        self.H1_rates: list[float] = [0]

    def tabulate_errors(
        self, n_dofs: int, mesh_size: float, L2_error: float, H1_error: float
    ) -> None:
        self.dof_sizes.append(n_dofs)
        self.mesh_sizes.append(mesh_size)
        self.L2_errors.append(L2_error)
        self.H1_errors.append(H1_error)

    def plot_errors(
        self,
        show: bool = True,
        save: bool = False,
        filename: str | None = None,
    ) -> None:
        plt.figure()
        ax: plt.Axes = plt.axes()
        ax.loglog(self.mesh_sizes, self.L2_errors, label="L2")
        ax.loglog(self.mesh_sizes, self.H1_errors, label="H1")
        plt.legend()
        if save:
            if filename is not None:
                plt.savefig(filename)
            else:
                plt.savefig("errors.png")
        if show:
            plt.show()
            plt.close()

    def compute_error_rates(self) -> None:
        for i in range(len(self.mesh_sizes) - 1):
            self.L2_rates.append(
                np.log(self.L2_errors[i + 1] / self.L2_errors[i])
                / np.log(self.mesh_sizes[i + 1] / self.mesh_sizes[i])
            )
            self.H1_rates.append(
                np.log(self.H1_errors[i + 1] / self.H1_errors[i])
                / np.log(self.mesh_sizes[i + 1] / self.mesh_sizes[i])
            )

    def print_errors(self) -> None:
        self.compute_error_rates()
        for i in range(len(self.mesh_sizes)):
            print(
                "dofs:",
                self.dof_sizes[i],
                "\th:",
                self.mesh_sizes[i],
                "\tL2:",
                self.L2_errors[i],
                "\tL2rate:",
                self.L2_rates[i],
                "\tH1:",
                self.H1_errors[i],
                "\tH1rate:",
                self.H1_rates[i],
            )

    def write_csv(self, filename_csv: str | None = None) -> None:
        data: dict = {
            "dofs": self.dof_sizes,
            "h": self.mesh_sizes,
            "L2": self.L2_errors,
            "rateL2": self.L2_rates,
            "H1": self.H1_errors,
            "rateH1": self.H1_rates,
        }
        df: pd.DataFrame = pd.DataFrame(data)
        for key in df.keys():
            df[key] = df[key].map("{:.3E}".format)
        if filename_csv is not None:
            df.to_csv(filename_csv, index=False)
        else:
            df.to_csv("errors.csv", index=False)


def problem_1():
    print("problem 1")

    def exact_value(x: np.ndarray) -> float:
        return np.cos(np.pi * x[0]) * np.cos(3 * np.pi * x[1])

    def exact_gradient(x: np.ndarray) -> np.ndarray:
        return -np.pi * np.array(
            [
                np.sin(np.pi * x[0]) * np.cos(3 * np.pi * x[1]),
                3 * np.cos(np.pi * x[0]) * np.sin(3 * np.pi * x[1]),
            ]
        )

    exact_solution: ExactSolution = ExactSolution(exact_value, exact_gradient)

    def f(x: np.ndarray) -> float:
        return (5 + 10 * np.pi**2) * np.cos(np.pi * x[0]) * np.cos(3 * np.pi * x[1])

    def g(_: np.ndarray | None = None) -> float:
        return 0

    q: float = 5
    problem_data: ProblemData = ProblemData(f, g, q)

    domain: Domain = make_square()
    max_mesh_size = 0.1
    mesh: tr.MeshInfo = triangulate_domain(domain, max_mesh_size)
    mesh_size: float = compute_mesh_size(mesh)
    solution: np.ndarray = solve(mesh, problem_data)

    L2_error: float = compute_L2_error(mesh, solution, exact_solution)
    H1_error: float = compute_H1_error(mesh, solution, exact_solution)

    error_table: ErrorTable = ErrorTable(len(solution), mesh_size, L2_error, H1_error)

    for _ in range(2):
        max_mesh_size /= 2
        mesh = refine_mesh(mesh, max_mesh_size)
        mesh_size: float = compute_mesh_size(mesh)
        solution = solve(mesh, problem_data)

        L2_error: float = compute_L2_error(mesh, solution, exact_solution)
        H1_error: float = compute_H1_error(mesh, solution, exact_solution)

        error_table.tabulate_errors(len(solution), mesh_size, L2_error, H1_error)

    plot_solution(mesh, solution, show=False, save=True, filename="problem_1_plot.png")
    error_table.print_errors()
    error_table.plot_errors(show=False, save=True, filename="problem_1_errors.png")
    error_table.write_csv("problem_1_errors.csv")


def problem_2():  # FIXME
    print("problem 2 q = 1")

    def f(_: np.ndarray | None = None) -> float:
        return 1

    def g(_: np.ndarray | None = None) -> float:
        return 1

    q: float = 1
    problem_data: ProblemData = ProblemData(f, g, q)

    domain: Domain = make_polygon()
    max_mesh_size = 0.1
    mesh: tr.MeshInfo = triangulate_domain(domain, max_mesh_size)
    solution: np.ndarray = solve(mesh, problem_data)

    plot_solution(
        mesh, solution, show=False, save=True, filename="problem_2_q_1_plot_0.png"
    )

    for i in [1, 2]:
        max_mesh_size /= 2
        mesh = refine_mesh(mesh, max_mesh_size)
        solution = solve(mesh, problem_data)
        plot_solution(
            mesh,
            solution,
            show=False,
            save=True,
            filename="problem_2_q_1_plot_" + str(i) + ".png",
        )

    print("problem 2 q = 0")

    q: float = 0
    problem_data: ProblemData = ProblemData(f, g, q)

    domain: Domain = make_polygon()
    max_mesh_size = 0.1
    mesh: tr.MeshInfo = triangulate_domain(domain, max_mesh_size)
    solution: np.ndarray = solve(mesh, problem_data)

    plot_solution(
        mesh, solution, show=False, save=True, filename="problem_2_q_0_plot_0.png"
    )

    for i in [1, 2]:
        max_mesh_size /= 2
        mesh = refine_mesh(mesh, max_mesh_size)
        solution = solve(mesh, problem_data)
        plot_solution(
            mesh,
            solution,
            show=False,
            save=True,
            filename="problem_2_q_0_plot_" + str(i) + ".png",
        )


def problem_3():
    print("problem 3 manufactured solution")

    def exact_value(x: np.ndarray) -> float:
        return np.cos(np.pi * x[0]) * np.cos(3 * np.pi * x[1])

    def exact_gradient(x: np.ndarray) -> np.ndarray:
        return -np.pi * np.array(
            [
                np.sin(np.pi * x[0]) * np.cos(3 * np.pi * x[1]),
                3 * np.cos(np.pi * x[0]) * np.sin(3 * np.pi * x[1]),
            ]
        )

    def exact_laplacian(x: np.ndarray) -> float:
        return -10 * np.pi**2 * exact_value(x)

    exact_solution: ExactSolution = ExactSolution(exact_value, exact_gradient)

    q: float = 1

    def f0(x: np.ndarray) -> float:
        return q * exact_value(x) - exact_laplacian(x)

    def g0(x: np.ndarray) -> float:
        if x[0] == 0:
            return np.dot(exact_gradient(x), np.array([-1, 0]))
        if x[0] == 0.25 and 0.25 <= x[1] <= 0.75:
            return np.dot(exact_gradient(x), np.array([1, 0]))
        if x[0] == 0.75 and 0.25 <= x[1] <= 0.75:
            return np.dot(exact_gradient(x), np.array([-1, 0]))
        if x[0] == 1:
            return np.dot(exact_gradient(x), np.array([1, 0]))

        if x[1] == 0:
            return np.dot(exact_gradient(x), np.array([0, -1]))
        if x[1] == 0.25 and 0.25 <= x[0] <= 0.75:
            return np.dot(exact_gradient(x), np.array([0, 1]))
        if x[1] == 0.75 and 0.25 <= x[0] <= 0.75:
            return np.dot(exact_gradient(x), np.array([0, -1]))
        if x[1] == 1:
            return np.dot(exact_gradient(x), np.array([0, 1]))

        return 0

    problem_data: ProblemData = ProblemData(f0, g0, q)

    domain: Domain = make_square_with_hole()
    max_mesh_size = 0.1
    mesh: tr.MeshInfo = triangulate_domain(domain, max_mesh_size)
    mesh_size: float = compute_mesh_size(mesh)
    solution: np.ndarray = solve(mesh, problem_data)

    L2_error: float = compute_L2_error(mesh, solution, exact_solution)
    H1_error: float = compute_H1_error(mesh, solution, exact_solution)

    error_table: ErrorTable = ErrorTable(len(solution), mesh_size, L2_error, H1_error)

    for _ in range(2):
        max_mesh_size /= 2
        mesh = refine_mesh(mesh, max_mesh_size)
        mesh_size: float = compute_mesh_size(mesh)
        solution = solve(mesh, problem_data)

        L2_error: float = compute_L2_error(mesh, solution, exact_solution)
        H1_error: float = compute_H1_error(mesh, solution, exact_solution)

        error_table.tabulate_errors(len(solution), mesh_size, L2_error, H1_error)

    plot_solution(
        mesh, solution, show=False, save=True, filename="problem_3_manufactured.png"
    )
    error_table.print_errors()
    error_table.plot_errors(
        show=False, save=True, filename="problem_3_manufactured_errors.png"
    )
    error_table.write_csv("problem_3_errors.csv")

    print("problem 3 unknown solution")

    exact_solution: ExactSolution = ExactSolution(exact_value, exact_gradient)

    q: float = 1

    def f(x: np.ndarray) -> float:
        return x[0] * x[1]

    def g(_: np.ndarray) -> float:
        return 1

    problem_data: ProblemData = ProblemData(f, g, q)

    domain: Domain = make_square_with_hole()
    max_mesh_size = 0.1
    mesh: tr.MeshInfo = triangulate_domain(domain, max_mesh_size)
    mesh_size: float = compute_mesh_size(mesh)
    solution: np.ndarray = solve(mesh, problem_data)

    for _ in range(2):
        max_mesh_size /= 2
        mesh = refine_mesh(mesh, max_mesh_size)
        mesh_size: float = compute_mesh_size(mesh)
        solution = solve(mesh, problem_data)

    plot_solution(mesh, solution, show=False, save=True, filename="problem_3_plot.png")


if __name__ == "__main__":
    problem_1()
    problem_2()
    problem_3()
