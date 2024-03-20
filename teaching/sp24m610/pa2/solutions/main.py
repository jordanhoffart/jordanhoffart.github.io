import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

q = 200
D = 8.8e7
l = 50


def T(x0, x1, x):
    return (x0 + x1) / 2 + (x1 - x0) / 2 * x


def dT(x0, x1):
    return (x1 - x0) / 2


bases = [lambda x: (1 - x) / 2, lambda x: (1 + x) / 2]
dbases = [-1 / 2, 1 / 2]

quad_pts = [0, -np.sqrt(3 / 5), np.sqrt(3 / 5)]
quad_wts = [8 / 9, 5 / 9, 5 / 9]


def compute_errors(nodes, Wh, W, dWdx):
    n_nodes = len(nodes)
    n_cells = n_nodes - 1
    cells = [[nodes[i], nodes[i + 1]] for i in range(n_cells)]
    global_indices_cells = [[i, i + 1] for i in range(n_cells)]

    L2_error = 0
    H1_error = 0
    Linf_error = 0

    for k in range(len(cells)):
        cell = cells[k]
        x0 = cell[0]
        x1 = cell[1]
        dT_cell = dT(x0, x1)
        global_indices_cell = global_indices_cells[k]

        for p in range(len(quad_pts)):
            xp = quad_pts[p]
            Wp = W(T(x0, x1, xp))
            dWdxp = dWdx(T(x0, x1, xp))
            wp = quad_wts[p]

            i0 = global_indices_cell[0]
            i1 = global_indices_cell[1]

            L2_error += (
                (Wp - Wh[i0] * bases[0](xp) - Wh[i1] * bases[1](xp)) ** 2 * dT_cell * wp
            )

            H1_error += (
                Wp - Wh[i0] * bases[0](xp) - Wh[i1] * bases[1](xp)
            ) ** 2 * dT_cell * wp + (
                dWdxp - Wh[i0] * dbases[0] / dT_cell - Wh[i1] * dbases[1] / dT_cell
            ) ** 2 * dT_cell * wp

            Linf_error = max(
                Linf_error, abs(Wp - Wh[i0] * bases[0](xp) - Wh[i1] * bases[1](xp))
            )

    return np.sqrt(L2_error), np.sqrt(H1_error), Linf_error


def problem_1():
    def solve(S, n_cells):
        def f(x):
            return q * x * (l - x) / 2 / D

        h = l / n_cells
        n_nodes = n_cells + 1
        nodes = [i * h for i in range(n_nodes)]
        cells = [[nodes[i], nodes[i + 1]] for i in range(n_cells)]
        global_indices_cells = [[i, i + 1] for i in range(n_cells)]

        n_dofs = n_nodes
        A = np.zeros((n_dofs, n_dofs))
        F = np.zeros(n_dofs)

        for k in range(len(cells)):
            cell = cells[k]
            x0 = cell[0]
            x1 = cell[1]
            dT_cell = dT(x0, x1)
            global_indices_cell = global_indices_cells[k]

            for p in range(len(quad_pts)):
                xp = quad_pts[p]
                wp = quad_wts[p]
                fp = f(T(x0, x1, xp))

                for i in range(len(bases)):
                    row = global_indices_cell[i]
                    test_ip = bases[i](xp)
                    dtest_i = dbases[i] / dT_cell

                    F[row] += fp * test_ip * dT_cell * wp

                    for j in range(len(bases)):
                        col = global_indices_cell[j]
                        trial_jp = bases[j](xp)
                        dtrial_j = dbases[j] / dT_cell

                        A[row, col] += S / D * trial_jp * test_ip * dT_cell * wp
                        A[row, col] += dtrial_j * dtest_i * dT_cell * wp

        A[0, :] = 0
        A[0, 0] = 1
        F[0] = 0

        A[-1, :] = 0
        A[-1, -1] = 1
        F[-1] = 0

        return h, nodes, np.linalg.solve(A, F)

    for S in [100, 1000, 10_000]:
        mesh_sizes = []
        L2_errors = []
        H1_errors = []
        Linfty_errors = []
        L2_rates = []
        H1_rates = []
        Linf_rates = []

        # errors_csv = open("problem_1_S_" + str(S) + ".csv", "w")
        # csv_writer = csv.writer(errors_csv)
        # csv_writer.writerow(["mesh size", "L2 Error", "H1 Error", "Linf Error"])

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.suptitle("Problem 1 S = " + str(S))

        a = S * l**2 / D
        b = q * l**4 / 2 / D

        def W(x):
            t = x / l
            return (
                b
                / a
                * (
                    -(t**2)
                    + t
                    - 2 / a
                    + 2
                    / a
                    / np.sinh(np.sqrt(a))
                    * (np.sinh(np.sqrt(a) * t) + np.sinh(np.sqrt(a) * (1 - t)))
                )
            )

        def dWdx(x):
            t = x / l
            dtdx = 1 / l
            dWdt = (
                b
                / a
                * (
                    -2 * t
                    + 1
                    + 2
                    / a
                    / np.sinh(np.sqrt(a))
                    * (
                        np.cosh(np.sqrt(a) * t) * np.sqrt(a)
                        + np.cosh(np.sqrt(a) * (1 - t)) * (-np.sqrt(a))
                    )
                )
            )
            return dWdt * dtdx

        n_cellss = [[25, 50], [100, 200]]
        for i in [0, 1]:
            for j in [0, 1]:
                n_cells = n_cellss[i][j]
                h, nodes, Wh = solve(S, n_cells)

                L2_error, H1_error, Linf_error = compute_errors(nodes, Wh, W, dWdx)

                mesh_sizes.append(h)
                L2_errors.append(L2_error)
                H1_errors.append(H1_error)
                Linfty_errors.append(Linf_error)

                axs[i, j].scatter(nodes, Wh, label="h = " + str(h), color="red")
                axs[i, j].plot(
                    nodes, [W(node) for node in nodes], label="exact", color="black"
                )
                axs[i, j].legend()

                if i == 0 and j == 0:
                    Linf_rates.append(0)
                    L2_rates.append(0)
                    H1_rates.append(0)
                    continue

                Linf_rate = np.log(Linfty_errors[-2] / Linfty_errors[-1]) / np.log(
                    mesh_sizes[-2] / mesh_sizes[-1]
                )
                L2_rate = np.log(L2_errors[-2] / L2_errors[-1]) / np.log(
                    mesh_sizes[-2] / mesh_sizes[-1]
                )
                H1_rate = np.log(H1_errors[-2] / H1_errors[-1]) / np.log(
                    mesh_sizes[-2] / mesh_sizes[-1]
                )

                Linf_rates.append(Linf_rate)
                L2_rates.append(L2_rate)
                H1_rates.append(H1_rate)

        # errors_csv.close()
        error_data = {
            "mesh size": mesh_sizes,
            "L2 error": L2_errors,
            "H1 error": H1_errors,
            "Linf error": Linfty_errors,
            "L2 rate": L2_rates,
            "H1 rate": H1_rates,
            "Linf rate": Linf_rates,
        }
        error_table = pd.DataFrame(error_data)
        error_table.to_csv(
            "problem_1_S_" + str(S) + ".csv", index=False, float_format="%.2E"
        )
        error_table.to_latex(
            "problem_1_S_" + str(S) + "_errors.tex", index=False, float_format="%.2E"
        )

        plt.savefig("problem_1_S_" + str(S) + ".png")

        fig, ax = plt.subplots()
        plt.title("Problem 1 S = " + str(S) + " errors")
        plt.xlabel("mesh size")
        plt.ylabel("error")
        plt.scatter(mesh_sizes[1:], L2_errors[1:])
        plt.scatter(mesh_sizes[1:], H1_errors[1:])
        plt.scatter(mesh_sizes[1:], Linfty_errors[1:])
        plt.plot(mesh_sizes[1:], L2_errors[1:], label="L2")
        plt.plot(mesh_sizes[1:], H1_errors[1:], label="H1")
        plt.plot(mesh_sizes[1:], Linfty_errors[1:], label="Linfty")
        plt.yscale("log")
        plt.xscale("log")
        ax.legend()
        plt.savefig("problem_1_S_" + str(S) + "_errors.png")


def problem_2():
    def solve(S, n_cells):
        def f(x):
            return q / 2 / D

        h = l / n_cells
        n_nodes = n_cells + 1
        nodes = [i * h for i in range(n_nodes)]
        cells = [[nodes[i], nodes[i + 1]] for i in range(n_cells)]
        global_indices_cells = [[i, i + 1] for i in range(n_cells)]

        n_dofs = n_nodes
        A = np.zeros((n_dofs, n_dofs))
        F = np.zeros(n_dofs)

        for k in range(len(cells)):
            cell = cells[k]
            x0 = cell[0]
            x1 = cell[1]
            dT_cell = dT(x0, x1)
            global_indices_cell = global_indices_cells[k]

            for p in range(len(quad_pts)):
                xp = quad_pts[p]
                wp = quad_wts[p]
                fp = f(T(x0, x1, xp))

                for i in range(len(bases)):
                    row = global_indices_cell[i]
                    test_ip = bases[i](xp)
                    dtest_i = dbases[i] / dT_cell

                    F[row] += fp * test_ip * dT_cell * wp

                    for j in range(len(bases)):
                        col = global_indices_cell[j]
                        trial_jp = bases[j](xp)
                        dtrial_j = dbases[j] / dT_cell

                        A[row, col] += S / D * trial_jp * test_ip * dT_cell * wp
                        A[row, col] += dtrial_j * dtest_i * dT_cell * wp

        A[0, :] = 0
        A[0, 0] = 1
        F[0] = 0

        A[-1, :] = 0
        A[-1, -1] = 1
        F[-1] = 0

        return h, nodes, np.linalg.solve(A, F)

    for S in [100, 1000, 10_000]:
        mesh_sizes = []
        L2_errors = []
        H1_errors = []
        Linfty_errors = []
        L2_rates = []
        H1_rates = []
        Linf_rates = []

        # errors_csv = open("problem_1_S_" + str(S) + ".csv", "w")
        # csv_writer = csv.writer(errors_csv)
        # csv_writer.writerow(["mesh size", "L2 Error", "H1 Error", "Linf Error"])

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.suptitle("Problem 2 S = " + str(S))

        a = S * l**2 / D
        Q = q * l**2 / 2 / D

        def W(x):
            t = x / l
            return (
                Q
                / a
                * (
                    1
                    - 1
                    / np.sinh(np.sqrt(a))
                    * (np.sinh(np.sqrt(a) * t) + np.sinh(np.sqrt(a) * (1 - t)))
                )
            )

        def dWdx(x):
            t = x / l
            dtdx = 1 / l
            dWdt = (
                Q
                / a
                * (
                    -1
                    / np.sinh(np.sqrt(a))
                    * (
                        np.cosh(np.sqrt(a) * t) * np.sqrt(a)
                        + np.cosh(np.sqrt(a) * (1 - t)) * (-np.sqrt(a))
                    )
                )
            )
            return dWdt * dtdx

        n_cellss = [[25, 50], [100, 200]]
        for i in [0, 1]:
            for j in [0, 1]:
                n_cells = n_cellss[i][j]
                h, nodes, Wh = solve(S, n_cells)

                L2_error, H1_error, Linf_error = compute_errors(nodes, Wh, W, dWdx)

                mesh_sizes.append(h)
                L2_errors.append(L2_error)
                H1_errors.append(H1_error)
                Linfty_errors.append(Linf_error)

                axs[i, j].scatter(nodes, Wh, label="h = " + str(h), color="red")
                axs[i, j].plot(
                    nodes, [W(node) for node in nodes], label="exact", color="black"
                )
                axs[i, j].legend()

                if i == 0 and j == 0:
                    Linf_rates.append(0)
                    L2_rates.append(0)
                    H1_rates.append(0)
                    continue

                Linf_rate = np.log(Linfty_errors[-2] / Linfty_errors[-1]) / np.log(
                    mesh_sizes[-2] / mesh_sizes[-1]
                )
                L2_rate = np.log(L2_errors[-2] / L2_errors[-1]) / np.log(
                    mesh_sizes[-2] / mesh_sizes[-1]
                )
                H1_rate = np.log(H1_errors[-2] / H1_errors[-1]) / np.log(
                    mesh_sizes[-2] / mesh_sizes[-1]
                )

                Linf_rates.append(Linf_rate)
                L2_rates.append(L2_rate)
                H1_rates.append(H1_rate)

        # errors_csv.close()
        error_data = {
            "mesh size": mesh_sizes,
            "L2 error": L2_errors,
            "H1 error": H1_errors,
            "Linf error": Linfty_errors,
            "L2 rate": L2_rates,
            "H1 rate": H1_rates,
            "Linf rate": Linf_rates,
        }
        error_table = pd.DataFrame(error_data)
        error_table.to_csv(
            "problem_2_S_" + str(S) + ".csv", index=False, float_format="%.2E"
        )
        error_table.to_latex(
            "problem_2_S_" + str(S) + "_errors.tex", index=False, float_format="%.2E"
        )

        plt.savefig("problem_2_S_" + str(S) + ".png")

        fig, ax = plt.subplots()
        plt.title("Problem 2 S = " + str(S) + " errors")
        plt.xlabel("mesh size")
        plt.ylabel("error")
        plt.scatter(mesh_sizes[1:], L2_errors[1:])
        plt.scatter(mesh_sizes[1:], H1_errors[1:])
        plt.scatter(mesh_sizes[1:], Linfty_errors[1:])
        plt.plot(mesh_sizes[1:], L2_errors[1:], label="L2")
        plt.plot(mesh_sizes[1:], H1_errors[1:], label="H1")
        plt.plot(mesh_sizes[1:], Linfty_errors[1:], label="Linfty")
        plt.yscale("log")
        plt.xscale("log")
        ax.legend()
        plt.savefig("problem_2_S_" + str(S) + "_errors.png")


def problem_3():
    def f1(x):
        return q * x * (l - x) / 2 / D

    def f2(x):
        return q / 2 / D

    def solve(S, n_cells, f):
        h = l / n_cells
        n_nodes = n_cells + 1
        nodes = [i * h for i in range(n_nodes)]
        cells = [[nodes[i], nodes[i + 1]] for i in range(n_cells)]
        global_indices_cells = [[i, i + 1] for i in range(n_cells)]

        n_dofs = n_nodes
        A = np.zeros((n_dofs, n_dofs))
        F = np.zeros(n_dofs)

        for k in range(len(cells)):
            cell = cells[k]
            x0 = cell[0]
            x1 = cell[1]
            dT_cell = dT(x0, x1)
            global_indices_cell = global_indices_cells[k]

            for p in range(len(quad_pts)):
                xp = quad_pts[p]
                wp = quad_wts[p]
                fp = f(T(x0, x1, xp))

                for i in range(len(bases)):
                    row = global_indices_cell[i]
                    test_ip = bases[i](xp)
                    dtest_i = dbases[i] / dT_cell

                    F[row] += fp * test_ip * dT_cell * wp

                    for j in range(len(bases)):
                        col = global_indices_cell[j]
                        trial_jp = bases[j](xp)
                        dtrial_j = dbases[j] / dT_cell

                        A[row, col] += S / D * trial_jp * test_ip * dT_cell * wp
                        A[row, col] += dtrial_j * dtest_i * dT_cell * wp

        A[0, :] = 0
        A[0, 0] = 1
        F[0] = 0

        return h, nodes, np.linalg.solve(A, F)

    fi_names = ["non_constant_rhs", "constant_rhs"]
    fi_plot_names = ["Non-constant rhs", "Constant rhs"]

    for fi, f in enumerate([f1, f2]):
        for S in [100, 1000, 10_000]:
            mesh_sizes = []
            L2_errors = []
            H1_errors = []
            Linfty_errors = []
            L2_rates = []
            H1_rates = []
            Linf_rates = []

            # errors_csv = open("problem_1_S_" + str(S) + ".csv", "w")
            # csv_writer = csv.writer(errors_csv)
            # csv_writer.writerow(["mesh size", "L2 Error", "H1 Error", "Linf Error"])

            fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
            fig.suptitle("Problem 3 " + fi_plot_names[fi] + " S = " + str(S))

            a = S * l**2 / D
            b = q * l**4 / 2 / D

            def W1(x):
                t = x / l
                return (
                    b
                    / a
                    * (
                        -(t**2)
                        + t
                        - 2 / a
                        + 1
                        / a
                        / np.cosh(np.sqrt(a))
                        * (
                            np.sqrt(a) * np.sinh(np.sqrt(a) * t)
                            + 2 * np.cosh(np.sqrt(a) * (1 - t))
                        )
                    )
                )

            def dW1dx(x):
                t = x / l
                dtdx = 1 / l
                dWdt = (
                    b
                    / a
                    * (
                        -2 * t
                        + 1
                        + 1
                        / a
                        / np.cosh(np.sqrt(a))
                        * (
                            a * np.cosh(np.sqrt(a) * t)
                            + np.sinh(np.sqrt(a) * (1 - t)) * (-2 * np.sqrt(a))
                        )
                    )
                )
                return dWdt * dtdx

            def W2(x):
                Q = q * l**2 / 2 / D
                t = x / l
                return (
                    Q
                    / a
                    * (
                        1
                        + np.sinh(np.sqrt(a))
                        / np.cosh(np.sqrt(a))
                        * np.sinh(np.sqrt(a) * t)
                        - np.cosh(np.sqrt(a) * t)
                    )
                )

            def dW2dx(x):
                t = x / l
                dtdx = 1 / l
                Q = q * l**2 / 2 / D
                dWdt = (
                    Q
                    / a
                    * (
                        np.sqrt(a)
                        * np.sinh(np.sqrt(a))
                        / np.cosh(np.sqrt(a))
                        * np.cosh(np.sqrt(a) * t)
                        - np.sinh(np.sqrt(a) * t) * np.sqrt(a)
                    )
                )
                return dWdt * dtdx

            Ws = [W1, W2]
            dWdxs = [dW1dx, dW2dx]

            n_cellss = [[25, 50], [100, 200]]
            for i in [0, 1]:
                for j in [0, 1]:
                    n_cells = n_cellss[i][j]
                    h, nodes, Wh = solve(S, n_cells, f)

                    L2_error, H1_error, Linf_error = compute_errors(
                        nodes, Wh, Ws[fi], dWdxs[fi]
                    )

                    mesh_sizes.append(h)
                    L2_errors.append(L2_error)
                    H1_errors.append(H1_error)
                    Linfty_errors.append(Linf_error)

                    axs[i, j].scatter(nodes, Wh, label="h = " + str(h), color="red")
                    axs[i, j].plot(
                        nodes,
                        [Ws[fi](node) for node in nodes],
                        label="exact",
                        color="black",
                    )
                    axs[i, j].legend()

                    if i == 0 and j == 0:
                        Linf_rates.append(0)
                        L2_rates.append(0)
                        H1_rates.append(0)
                        continue

                    Linf_rate = np.log(Linfty_errors[-2] / Linfty_errors[-1]) / np.log(
                        mesh_sizes[-2] / mesh_sizes[-1]
                    )
                    L2_rate = np.log(L2_errors[-2] / L2_errors[-1]) / np.log(
                        mesh_sizes[-2] / mesh_sizes[-1]
                    )
                    H1_rate = np.log(H1_errors[-2] / H1_errors[-1]) / np.log(
                        mesh_sizes[-2] / mesh_sizes[-1]
                    )

                    Linf_rates.append(Linf_rate)
                    L2_rates.append(L2_rate)
                    H1_rates.append(H1_rate)

            # errors_csv.close()
            error_data = {
                "mesh size": mesh_sizes,
                "L2 error": L2_errors,
                "H1 error": H1_errors,
                "Linf error": Linfty_errors,
                "L2 rate": L2_rates,
                "H1 rate": H1_rates,
                "Linf rate": Linf_rates,
            }
            error_table = pd.DataFrame(error_data)
            error_table.to_csv(
                "problem_3_" + fi_names[fi] + "_S_" + str(S) + ".csv",
                index=False,
                float_format="%.2E",
            )
            error_table.to_latex(
                "problem_3_" + fi_names[fi] + "_S_" + str(S) + "_errors.tex",
                index=False,
                float_format="%.2E",
            )

            plt.savefig("problem_3_" + fi_names[fi] + "_S_" + str(S) + ".png")

            fig, ax = plt.subplots()
            plt.title("Problem 3 " + fi_plot_names[fi] + " S = " + str(S) + " errors")
            plt.xlabel("mesh size")
            plt.ylabel("error")
            plt.scatter(mesh_sizes[1:], L2_errors[1:])
            plt.scatter(mesh_sizes[1:], H1_errors[1:])
            plt.scatter(mesh_sizes[1:], Linfty_errors[1:])
            plt.plot(mesh_sizes[1:], L2_errors[1:], label="L2")
            plt.plot(mesh_sizes[1:], H1_errors[1:], label="H1")
            plt.plot(mesh_sizes[1:], Linfty_errors[1:], label="Linfty")
            plt.yscale("log")
            plt.xscale("log")
            ax.legend()
            plt.savefig("problem_3_" + fi_names[fi] + "_S_" + str(S) + "_errors.png")


def problem_4():
    def solve(n_cells):
        def k(x):
            if 0 <= x < np.pi / 6:
                return 1
            if np.pi / 6 <= x < np.pi / 4:
                return 2
            if np.pi / 4 <= x <= 1:
                return 3
            return 0

        def v(x):
            return (4 / np.pi + 3 / 2) * x

        def f(x):
            return -(4 / np.pi + 3 / 2) * k(x)

        n_nodes = int((n_cells + 1) / 3)
        discontinuities = [0, np.pi / 6, np.pi / 4, 1]
        nodes_pieces = [
            [
                discontinuities[j]
                + i * (discontinuities[j + 1] - discontinuities[j]) / (n_nodes - 1)
                for i in range(n_nodes - 1)
            ]
            for j in range(len(discontinuities) - 1)
        ]
        nodes = []
        for j in range(len(discontinuities) - 1):
            for i in range(len(nodes_pieces[j])):
                nodes.append(nodes_pieces[j][i])
        nodes.append(1)
        print(nodes)
        n_nodes = len(nodes)
        h = 0
        for i in range(n_nodes - 1):
            h = max(h, nodes[i + 1] - nodes[i])
        cells = [[nodes[i], nodes[i + 1]] for i in range(n_nodes - 1)]
        global_indices_cells = [[i, i + 1] for i in range(n_nodes - 1)]

        n_dofs = n_nodes
        A = np.zeros((n_dofs, n_dofs))
        F = np.zeros(n_dofs)

        for ik in range(len(cells)):
            cell = cells[ik]
            x0 = cell[0]
            x1 = cell[1]
            dT_cell = dT(x0, x1)
            global_indices_cell = global_indices_cells[ik]

            for p in range(len(quad_pts)):
                xp = quad_pts[p]
                wp = quad_wts[p]
                fp = f(T(x0, x1, xp))
                kp = k(T(x0, x1, xp))

                for i in range(len(bases)):
                    row = global_indices_cell[i]
                    dtest_i = dbases[i] / dT_cell

                    F[row] += fp * dtest_i * dT_cell * wp

                    for j in range(len(bases)):
                        col = global_indices_cell[j]
                        dtrial_j = dbases[j] / dT_cell

                        A[row, col] += kp * dtrial_j * dtest_i * dT_cell * wp

        A[0, :] = 0
        A[0, 0] = 1
        F[0] = 0

        A[-1, :] = 0
        A[-1, -1] = 1
        F[-1] = 0

        Wh = np.linalg.solve(A, F)
        Uh = [Wh[i] + v(nodes[i]) for i in range(len(nodes))]

        return h, nodes, Uh

    mesh_sizes = []
    L2_errors = []
    H1_errors = []
    Linfty_errors = []
    L2_rates = []
    H1_rates = []
    Linf_rates = []

    # errors_csv = open("problem_1_S_" + str(S) + ".csv", "w")
    # csv_writer = csv.writer(errors_csv)
    # csv_writer.writerow(["mesh size", "L2 Error", "H1 Error", "Linf Error"])

    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle("Problem 4")

    def W(x):
        if 0 <= x < np.pi / 6:
            return 12 / np.pi * x
        if np.pi / 6 <= x < np.pi / 4:
            return 6 / np.pi * x + 1
        if np.pi / 4 <= x <= 1:
            return 4 / np.pi * x + 3 / 2
        return 0

    def dWdx(x):
        if 0 <= x < np.pi / 6:
            return 12 / np.pi
        if np.pi / 6 <= x < np.pi / 4:
            return 6 / np.pi
        if np.pi / 4 <= x <= 1:
            return 4 / np.pi
        return 0

    n_cellss = [[25, 50], [100, 200]]
    for i in [0, 1]:
        for j in [0, 1]:
            n_cells = n_cellss[i][j]
            h, nodes, Wh = solve(n_cells)

            L2_error, H1_error, Linf_error = compute_errors(nodes, Wh, W, dWdx)

            mesh_sizes.append(h)
            L2_errors.append(L2_error)
            H1_errors.append(H1_error)
            Linfty_errors.append(Linf_error)

            axs[i, j].scatter(nodes, Wh, label="h = {:.2e}".format(h), color="red")
            axs[i, j].plot(
                nodes, [W(node) for node in nodes], label="exact", color="black"
            )
            axs[i, j].legend()

            continue

            if i == 0 and j == 0:
                Linf_rates.append(0)
                L2_rates.append(0)
                H1_rates.append(0)
                continue

            Linf_rate = np.log(Linfty_errors[-2] / Linfty_errors[-1]) / np.log(
                mesh_sizes[-2] / mesh_sizes[-1]
            )
            L2_rate = np.log(L2_errors[-2] / L2_errors[-1]) / np.log(
                mesh_sizes[-2] / mesh_sizes[-1]
            )
            H1_rate = np.log(H1_errors[-2] / H1_errors[-1]) / np.log(
                mesh_sizes[-2] / mesh_sizes[-1]
            )

            Linf_rates.append(Linf_rate)
            L2_rates.append(L2_rate)
            H1_rates.append(H1_rate)

    # errors_csv.close()
    error_data = {
        "mesh size": mesh_sizes,
        "L2 error": L2_errors,
        "H1 error": H1_errors,
        "Linf error": Linfty_errors,
        # "L2 rate": L2_rates,
        # "H1 rate": H1_rates,
        # "Linf rate": Linf_rates,
    }
    error_table = pd.DataFrame(error_data)
    error_table.to_csv("problem_4.csv", index=False, float_format="%.2E")
    error_table.to_latex("problem_4_errors.tex", index=False, float_format="%.2E")

    plt.savefig("problem_4.png")

    return
    fig, ax = plt.subplots()
    plt.title("Problem 4 errors")
    plt.xlabel("mesh size")
    plt.ylabel("error")
    plt.scatter(mesh_sizes[1:], L2_errors[1:])
    plt.scatter(mesh_sizes[1:], H1_errors[1:])
    plt.scatter(mesh_sizes[1:], Linfty_errors[1:])
    plt.plot(mesh_sizes[1:], L2_errors[1:], label="L2")
    plt.plot(mesh_sizes[1:], H1_errors[1:], label="H1")
    plt.plot(mesh_sizes[1:], Linfty_errors[1:], label="Linfty")
    plt.yscale("log")
    plt.xscale("log")
    ax.legend()
    plt.savefig("problem_4_errors.png")


def main():
    # problem_1()
    # problem_2()
    # problem_3()
    problem_4()


if __name__ == "__main__":
    main()
