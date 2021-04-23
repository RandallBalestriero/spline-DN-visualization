import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
from . import geometry
from . import utils

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm

plt.style.use("/home/vrael/DeeSect/presentation.mplstyle")


def draw_layer_paths(ax, paths, colors="k", **line_kwargs):
    if type(paths) == list and type(colors) == list:
        assert len(paths) == len(colors)
        for p, c in zip(paths, colors):
            draw_layer_paths(ax, p, c, **line_kwargs)
    elif type(paths) == list and type(colors) != list:
        for p in paths:
            draw_layer_paths(ax, p, colors, **line_kwargs)
    else:
        if paths.shape[1] == 2:
            ax.plot(
                paths[:, 0],
                paths[:, 1],
                c=colors,
                **line_kwargs,
            )
        elif paths.shape[1] == 3:
            ax.plot(
                paths[:-1, 0],
                paths[:-1, 1],
                paths[:-1, 2],
                c=colors,
                **line_kwargs,
            )


def codes_to_colors(codes, cmap="Spectral", n_colors=10):
    unique_codes = set(map(tuple, codes))
    unique_values = np.linspace(0, 1, min(n_colors, len(unique_codes)))[
        np.arange(len(unique_codes)) % n_colors
    ]
    code_to_value = {a: t for a, t in zip(unique_codes, unique_values)}

    if type(cmap) == str:
        cmap = matplotlib.cm.get_cmap(cmap)

    values = np.array([code_to_value[tuple(code)] for code in codes])
    colors = cmap(values)
    return values, colors


def pretty_plot(
    network,
    n_samples_x=100,
    n_samples_y=100,
    extent=[-2, 2, -2, 2],
    n_colors=10,
    cmap="Spectral",
    boundary_colors="k",
    figsize=(5, 5),
    colorize=True,
    color_values=None,
    color_mapping=None,
    partition_figure_name="partition_figure",
    name_output_space=None,
    kwargs_output_space_plot=None,
):
    x = np.linspace(extent[0], extent[1], n_samples_x)
    y = np.linspace(extent[2], extent[3], n_samples_y)

    meshgrid = np.meshgrid(x, y)
    dn_input = np.stack([meshgrid[0].reshape(-1), meshgrid[1].reshape(-1)], 1)

    feature_maps = network.preactivations(dn_input)
    codes = network.codes(dn_input)

    paths = utils.zero_set_paths(feature_maps=feature_maps, meshgrid=meshgrid)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    if colorize:
        values, colors = codes_to_colors(codes, cmap="Spectral", n_colors=10)
        ax.imshow(
            values.reshape(meshgrid[0].shape),
            cmap="Spectral",
            vmin=0,
            vmax=1,
            extent=extent,
        )

    # else:
    #     cmap = matplotlib.cm.get_cmap(cmap)

    #     if color_values is not None:
    #         values = color_values
    #     else:
    #         values = color_mapping(meshgrid_flat)

    #     colors = cmap(values)
    #     ax.pcolormesh(
    #         meshgrid[0],
    #         meshgrid[1],
    #         values.reshape(meshgrid[0].shape),
    #         cmap=cmap,
    #         alpha=1,
    #     )
    draw_layer_paths(ax, paths, colors=boundary_colors, zorder=1000)
    # surface = input_output_mapping(meshgrid_flat)
    plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    plt.savefig(partition_figure_name)
    plt.close()
    return

    mapped_paths = []
    for path in paths:
        mapped_paths.append([])
        for p in path:
            mapped_paths[-1].append(input_output_mapping(p.vertices[:-1]))

    if kwargs_output_space_plot is None:
        kwargs_output_space_plot = {}

    fig = plt.figure()
    if surface.shape[1] == 1:
        ax = fig.add_subplot(1, 1, 1, projection="3d", proj_type="persp")
        ax.plot_surface(
            meshgrid[0],
            meshgrid[1],
            surface.reshape(meshgrid[0].shape),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            facecolors=colors.reshape(meshgrid[0].shape + (4,)),
            shade=True,
        )
        make_3daxis_pretty(ax)
    elif surface.shape[1] == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.pcolormesh(
            surface[:, 0].reshape(meshgrid[0].shape),
            surface[:, 1].reshape(meshgrid[0].shape),
            values.reshape(meshgrid[0].shape),
            cmap=cmap,
            alpha=1,
        )
        plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)
    elif surface.shape[1] == 3:
        ax = fig.add_subplot(1, 1, 1, projection="3d", proj_type="persp")
        colors = colors.reshape(meshgrid[0].shape + (4,))
        colors[:, 3] = 1
        ax.plot_surface(
            surface[:, 0].reshape(meshgrid[0].shape),
            surface[:, 1].reshape(meshgrid[0].shape),
            surface[:, 2].reshape(meshgrid[0].shape),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            facecolors=colors,
            shade=False,
            alpha=0.8,
        )
        make_3daxis_pretty(ax)
    draw_layer_paths(
        ax,
        mapped_paths,
        # zorder=1000,
        linewidth=matplotlib.rcParams["lines.linewidth"] * 0.8,
    )

    if name_output_space is not None:
        plt.savefig(name_output_space)
        plt.close()

    plt.show()


def pretty_onelayer_partition(
    layer_W,
    layer_b,
    layer_alpha,
    n_samples_x=500,
    n_samples_y=500,
    extent=[-3, 3, -3, 3],
    meshgrid=None,
    n_colors=10,
    cmap="Spectral",
    name=None,
    with_power_diagram=True,
):
    def per_unit_mapping(x):
        h = x.dot(layer_W.T) + layer_b
        return h

    def input_code_mapping(x):
        h = x.dot(layer_W.T) + layer_b
        return (h > 0).astype("int32")

    paths, meshgrid, meshgrid_flat = partition_boundary_2d(
        per_unit_mapping,
        n_samples_x=n_samples_x,
        n_samples_y=n_samples_y,
        extent=extent,
        meshgrid=meshgrid,
    )

    codes = input_code_mapping(meshgrid_flat)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    values, colors = draw_colored_partition_from_codes(
        ax,
        codes,
        xy_shape=meshgrid[0].shape,
        n_colors=n_colors,
        cmap=cmap,
        extent=[
            meshgrid[0].min(),
            meshgrid[0].max(),
            meshgrid[1].min(),
            meshgrid[1].max(),
        ],
    )

    draw_layer_paths(ax, paths, zorder=1000)

    if with_power_diagram:
        mus, radii, colors = geometry.get_layer_PD(
            meshgrid_flat, layer_W, layer_b, layer_alpha, colors
        )

        for center, rad, color in zip(mus, radii, colors):

            fc = color.copy()
            fc[-1] = 0.3
            fc[:-1] += 0.08
            fc = np.clip(fc, 0, 1)

            ec = color.copy()
            ec[-1] = 0.8
            ec[:-1] -= 0.08
            ec = np.clip(ec, 0, 1)

            circle = matplotlib.patches.Circle(
                center, radius=rad, facecolor=fc, edgecolor=ec, zorder=1500
            )
            ax.add_artist(circle)
            ax.scatter(*center, c=[color], zorder=2000, edgecolor="k")

    ax.set_xlim([meshgrid[0].min(), meshgrid[0].max()])
    ax.set_ylim([meshgrid[1].min(), meshgrid[1].max()])
    plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)
    if name is not None:
        plt.xticks([])
        plt.yticks([])
        plt.savefig(name)
    else:
        plt.show()


def draw_partition(
    meshgrid,
    facevalues=None,
    facecolors=None,
    paths=None,
    subplot=[1, 1],
    pathcolors=None,
    paths_only=False,
    fig=None,
):
    if fig is None:
        fig = make_subplots(
            rows=1,
            cols=1,
            horizontal_spacing=0.01,
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    if facevalues is not None:
        if facecolors is not None:
            extra = {"colorscale": facecolors}
        else:
            extra = {}
        trace = go.Heatmap(
            x=meshgrid[0][0],
            y=meshgrid[1][:, 0],
            z=facevalues.reshape(meshgrid[0].shape),
            showscale=False,
            **extra,
        )

        fig.add_trace(trace, *subplot)

    if paths is None:
        return

    for layer in range(len(paths)):

        for unit in range(len(paths[layer])):
            color = (
                "#000000" if pathcolors is None else pathcolors[layer][unit]
            )
            trace = go.Scatter(
                x=paths[layer][unit][0],
                y=paths[layer][unit][1],
                line=dict(color=color, width=5),
                mode="lines",
                name="",
                showlegend=False,
            )
            fig.add_trace(trace, *subplot)

    fig.update_xaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        nticks=0,
        tickfont=dict(family="Helvetica, monospace", size=25, color="black"),
        range=[meshgrid[0].min(), meshgrid[0].max()],
        row=subplot[0],
        col=subplot[1],
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        nticks=0,
        tickfont=dict(family="Helvetica, monospace", size=25, color="black"),
        range=[meshgrid[1].min(), meshgrid[1].max()],
        row=subplot[0],
        col=subplot[1],
        scaleanchor="y",
        scaleratio=1,
    )
    return fig


def draw_surface(
    fig, x, y, z, subplot=[1, 1], scene="scene", clean=False, **extra_kwargs
):
    trace = go.Surface(
        x=x,
        y=y,
        z=z,
        lighting=dict(specular=0.3),
        scene=scene,
        **extra_kwargs,
    )
    fig.add_trace(trace, *subplot)

    camera = dict(eye=dict(x=1, y=-2.0, z=0.5))

    fig["layout"][scene].camera = camera
    if clean:
        fig["layout"][scene].xaxis.update(
            showline=False,
            title="",
            nticks=0,
            tickvals=[],
            backgroundcolor="rgb(255, 255, 255)",
        )
        fig["layout"][scene].yaxis.update(
            showline=False,
            title="",
            nticks=0,
            tickvals=[],
            backgroundcolor="rgb(255, 255, 255)",
        )
        fig["layout"][scene].zaxis.update(
            showline=False,
            title="",
            nticks=0,
            tickvals=[],
            backgroundcolor="rgb(255, 255, 255)",
        )
        return

    fig["layout"][scene].xaxis.update(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        title="",
        tickfont=dict(family="Helvetica, monospace", size=19, color="black"),
        nticks=5,
    )
    fig["layout"][scene].yaxis.update(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        title="",
        tickfont=dict(family="Helvetica, monospace", size=19, color="black"),
        nticks=5,
    )
    fig["layout"][scene].zaxis.update(
        showline=True,
        linewidth=4,
        linecolor="black",
        mirror=True,
        title="",
        tickfont=dict(family="Helvetica, monospace", size=19, color="black"),
        nticks=5,
    )


def output_space_pretty_plot(
    input_output_mapping,
    paths,
    meshgrid,
    meshgrid_flat,
    facevalues,
    facecolors,
    kwargs_output_space_plot=None,
):

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{}, {"type": "surface"}]],
        horizontal_spacing=0.01,
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")

    draw_partition(
        fig, meshgrid, facevalues, facecolors, paths=paths, subplot=[1, 1]
    )

    surface = input_output_mapping(meshgrid_flat)

    mapped_paths = []
    for path in paths:
        mapped_paths.append([])
        for p in path:
            mapped_paths[-1].append(input_output_mapping(p.vertices[:-1]))

    if surface.shape[1] == 1 or surface.shape[1] == 3:
        if surface.shape[1] == 1:
            x = meshgrid[0]
            y = meshgrid[1]
            z = surface.reshape(meshgrid[0].shape)
        else:
            x = surface[:, 0].reshape(meshgrid[0].shape)
            y = surface[:, 1].reshape(meshgrid[0].shape)
            z = surface[:, 2].reshape(meshgrid[0].shape)
        extra = {"showscale": False, "opacity": 0.8}
        if facevalues is not None:
            extra.update(
                {"surfacecolor": facevalues.reshape(meshgrid[0].shape)}
            )
        if facecolors is not None:
            if type(facecolors) == str:
                extra.update({"colorscale": facecolors})
            else:
                extra.update({"colorscale": list(zip(facevalues, facecolors))})
        draw_surface(fig, x, y, z, subplot=[1, 2], **extra)

    elif surface.shape[1] == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.pcolormesh(
            surface[:, 0].reshape(meshgrid[0].shape),
            surface[:, 1].reshape(meshgrid[0].shape),
            values.reshape(meshgrid[0].shape),
            cmap=cmap,
            alpha=1,
        )
        plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)
    else:
        raise RuntimeError

    for layerpaths in mapped_paths:
        for path in layerpaths:
            trace = go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                line=dict(color="#000000", width=8),
                mode="lines",
                name="",
                showlegend=False,
            )
            fig.add_trace(trace, 1, 2)

    fig.show()


def multisurface_pretty_plot(
    input_output_mapping,
    paths,
    meshgrid,
    meshgrid_flat,
    facecolors,
    kwargs_output_space_plot=None,
):

    N = len(input_output_mapping)
    fig = make_subplots(
        rows=1,
        cols=N,
        specs=[[{"type": "surface"}] * N],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.0,
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")

    for i in range(N):

        surface = input_output_mapping[i](meshgrid_flat)

        mapped_paths = []
        for path in paths[i]:
            mapped_paths.append([])
            for p in path:
                mapped_paths[-1].append(
                    input_output_mapping[i](p.vertices[:-1])
                )

        if surface.shape[1] == 1:
            x = meshgrid[0]
            y = meshgrid[1]
            z = surface.reshape(meshgrid[0].shape)
        else:
            x = surface[:, 0].reshape(meshgrid[0].shape)
            y = surface[:, 1].reshape(meshgrid[0].shape)
            z = surface[:, 2].reshape(meshgrid[0].shape)
        extra = {"showscale": False, "opacity": 0.8}
        if facecolors is not None:
            if type(facecolors) == str:
                extra.update({"colorscale": facecolors})
            else:
                extra.update({"colorscale": list(zip(facevalues, facecolors))})
        draw_surface(
            fig,
            x,
            y,
            z,
            subplot=[1, 1 + i],
            clean=True,
            scene="scene" + str(1 + i) if i else "scene",
            **extra,
        )

        for layerpaths in mapped_paths:
            for path in layerpaths:
                trace = go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    line=dict(color="#000000", width=8),
                    mode="lines",
                    name="",
                    showlegend=False,
                )
                fig.add_trace(trace, 1, 1 + i)

    fig.show()


def draw_mlp(MLP):

    node_trace = go.Scatter(
        x=np.concatenate(MLP.nodes_x),
        y=np.concatenate(MLP.nodes_y),
        mode="markers",
        marker=dict(
            size=20,
            line=dict(color="black", width=2),
        ),
        hoverinfo="text",
        name="nodes",
    )

    node_trace.marker.color = sum(MLP.nodes_color, [])
    node_trace.text = sum(MLP.nodes_text, [])

    edge_trace = go.Scatter(
        x=MLP.edges_x,
        y=MLP.edges_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
        name="edges",
    )

    figure = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            hovermode="closest",
        ),
    )
    figure.update_xaxes(visible=False, fixedrange=True)
    figure.update_yaxes(visible=False, fixedrange=True)

    # remove facet/subplot labels
    figure.update_layout(
        overwrite=True,
        plot_bgcolor="white",
        annotations=[
            dict(
                x=0,
                y=0.7,
                xref="x",
                yref="y",
                text="Input",
                ax=0,
                ay=0.0,
                font=dict(family="Open Sans", size=20),
            )
        ],
        legend_font=dict(family="Open Sans", size=20),
        margin=dict(l=5, r=5, t=5, b=5),
    )
    return figure


def create_dn(MLP):

    fig = draw_partition(
        meshgrid=MLP.meshgrid,
        paths=MLP.paths,
        pathcolors=MLP.nodes_color[1:],
    )

    fig2 = make_subplots(
        rows=1,
        cols=len(MLP.widths) - 1,
        horizontal_spacing=0.05,
    )
    fig.update_layout(
        plot_bgcolor="white",
        margin=dict(l=5, r=5, t=5, b=5),
    )
    fig2.update_layout(
        plot_bgcolor="white",
        margin=dict(l=5, r=5, t=5, b=5),
    )
    for l in range(len(MLP.widths) - 1):
        draw_partition(
            meshgrid=MLP.meshgrid,
            paths=[MLP.paths[l]],
            pathcolors=[MLP.nodes_color[1 + l]],
            fig=fig2,
            subplot=[1, 1 + l],
        )
    return fig, fig2
