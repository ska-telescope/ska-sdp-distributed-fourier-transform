#!/usr/bin/env python
# coding: utf-8

import math
import time
import itertools
import logging
import sys
from matplotlib import pylab

from fourier_algorithm import *

# Plot setup
from src.fourier_transform.utils import (
    mark_range,
    whole,
    test_accuracy_subgrid_to_facet,
    test_accuracy_facet_to_subgrid,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# Fixing seed of numpy random
numpy.random.seed(123456789)

pylab.rcParams["figure.figsize"] = 16, 4
pylab.rcParams["image.cmap"] = "viridis"

target_pars = {
    "W": 13.25,
    "fov": 0.75,
    "N": 1024,
    "Nx": 4,
    "yB_size": 256,
    "yN_size": 320,
    "yP_size": 512,
    "xA_size": 188,
    "xM_size": 256,
}

target_err = 1e-5

alpha = 0

# needed for degridding
W_steps = 32
Ws = numpy.arange(2, 22, 1 / W_steps)
res = 1024
normal = numpy.prod(numpy.arange(2 * alpha - 1, 0, -2, dtype=float))  # double factorial
pswfs = {
    W: anti_aliasing_function(res, alpha, numpy.pi * W / 2).real / normal for W in Ws
}

# Note: cell[-3] (defining `test_degrid_accuracy`) also has some extra
#       functions and variables copied from earlier parts


print("== Chosen configuration")
for n in ["W", "fov", "N", "Nx", "yB_size", "yN_size", "yP_size", "xA_size", "xM_size"]:
    exec(f"{n} = target_pars[n]")
    print(f"{n} = {target_pars[n]}")

print("\n== Relative coordinates")
xN = W / yN_size / 2
xM = xM_size / 2 / N
yN = yN_size / 2
xA = xA_size / 2 / N
yB = yB_size / 2
print("xN=%g xM=%g yN=%g xNyN=%g xA=%g" % (xN, xM, yN, xN * yN, xA))

print("\n== Derived values")
xN_size = N * W / yN_size
xM_yP_size = xM_size * yP_size // N
xMxN_yP_size = xM_yP_size + int(2 * numpy.ceil(xN_size * yP_size / N / 2))
assert (xM_size * yN_size) % N == 0
xM_yN_size = xM_size * yN_size // N

print(
    f"xN_size={xN_size:.1f} xM_yP_size={xM_yP_size}, xMxN_yP_size={xMxN_yP_size}, xM_yN_size={xM_yN_size}"
)
if fov is not None:
    nfacet = int(numpy.ceil(N * fov / yB_size))
    print(
        f"{nfacet}x{nfacet} facets for FoV of {fov} ({N * fov / nfacet / yB_size * 100}% efficiency)"
    )


# ## Calculate PSWF
# Calculate PSWF at the full required resolution (facet size)

pswf = anti_aliasing_function(yN_size, alpha, numpy.pi * W / 2).real
pswf /= numpy.prod(numpy.arange(2 * alpha - 1, 0, -2, dtype=float))  # double factorial

x = coordinates(N)
fx = N * coordinates(N)
n = ifft(pad_mid(pswf, N))
pylab.semilogy(
    coordinates(4 * int(xN_size)) * 4 * xN_size / N,
    extract_mid(numpy.abs(ifft(pad_mid(pswf, N))), 4 * int(xN_size)),
)
pylab.legend(["n"])
mark_range("$x_n$", -xN, xN)
pylab.xlim(-2 * int(xN_size) / N, (2 * int(xN_size) - 1) / N)
pylab.grid()
pylab.show()
# pylab.savefig("plot_n.png")
pylab.semilogy(coordinates(yN_size) * yN_size, pswf)
pylab.legend(["$\\mathcal{F}[n]$"])
mark_range("$y_B$", -yB, yB)
pylab.xlim(-N // 2, N // 2 - 1)
mark_range("$y_n$", -yN, yN)
pylab.grid()
pylab.show()
# pylab.savefig("plot_fn.png")

# Calculate actual work terms to use. We need both $n$ and $b$ in image space.
Fb = 1 / extract_mid(pswf, yB_size)
Fn = pswf[(yN_size // 2) % int(1 / 2 / xM) :: int(1 / 2 / xM)]
facet_m0_trunc = pswf * numpy.sinc(coordinates(yN_size) * xM_size / N * yN_size)
facet_m0_trunc = (
    xM_size
    * yP_size
    / N
    * extract_mid(ifft(pad_mid(facet_m0_trunc, yP_size)), xMxN_yP_size).real
)
pylab.semilogy(coordinates(xMxN_yP_size) / yP_size * xMxN_yP_size, facet_m0_trunc)
mark_range("xM", -xM, xM)
pylab.grid()
pylab.show()
# pylab.savefig("plot_xm.png")

## Layout subgrids + facets

nsubgrid = int(math.ceil(N / xA_size))
nfacet = int(math.ceil(N / yB_size))
print("%d subgrids, %d facets needed to cover" % (nsubgrid, nfacet))
subgrid_off = xA_size * numpy.arange(nsubgrid) + Nx
facet_off = yB_size * numpy.arange(nfacet)


assert whole(numpy.outer(subgrid_off, facet_off) / N)
assert whole(facet_off * xM_size / N)

# # Determine subgrid/facet offsets and the appropriate A/B masks for cutting them out. We are aiming for full coverage here: Every pixel is part of exactly one subgrid / facet.


subgrid_A = numpy.zeros((nsubgrid, xA_size), dtype=int)
subgrid_border = (
    subgrid_off + numpy.hstack([subgrid_off[1:], [N + subgrid_off[0]]])
) // 2
for i in range(nsubgrid):
    left = (subgrid_border[i - 1] - subgrid_off[i] + xA_size // 2) % N
    right = subgrid_border[i] - subgrid_off[i] + xA_size // 2
    assert left >= 0 and right <= xA_size, "xA not large enough to cover subgrids!"
    subgrid_A[i, left:right] = 1

facet_B = numpy.zeros((nfacet, yB_size), dtype=bool)
facet_split = numpy.array_split(range(N), nfacet)
facet_border = (facet_off + numpy.hstack([facet_off[1:], [N]])) // 2
for j in range(nfacet):
    left = (facet_border[j - 1] - facet_off[j] + yB_size // 2) % N
    right = facet_border[j] - facet_off[j] + yB_size // 2
    assert left >= 0 and right <= yB_size, "yB not large enough to cover facets!"
    facet_B[j, left:right] = 1


G = numpy.random.rand(N) - 0.5
subgrid, facet = make_subgrid_and_facet(
    G, nsubgrid, xA_size, subgrid_A, subgrid_off, nfacet, yB_size, facet_B, facet_off
)

# # With a few more slight optimisations we arrive at a compact representation for our algorithm. For reference, what we are computing here is:

dtype = numpy.complex128
xN_yP_size = xMxN_yP_size - xM_yP_size

print("Facet data:", facet.shape, facet.size)
nmbfs = facets_to_subgrid_1d(
    facet,
    nsubgrid,
    nfacet,
    xM_yN_size,
    Fb,
    Fn,
    yP_size,
    facet_m0_trunc,
    subgrid_off,
    N,
    xMxN_yP_size,
    xN_yP_size,
    xM_yP_size,
    dtype,
)

# - redistribution of nmbfs here -
print("Redistributed data:", nmbfs.shape, nmbfs.size)
approx_subgrid = reconstruct_subgrid_1d(
    nmbfs, xM_size, nfacet, facet_off, N, subgrid_A, xA_size, nsubgrid
)
print("Reconstructed subgrids:", approx_subgrid.shape, approx_subgrid.size)


# Let us look at the error terms:
fig = pylab.figure(figsize=(16, 8))
ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
err_sum = err_sum_img = 0
for i in range(nsubgrid):
    error = approx_subgrid[i] - subgrid[i]
    ax1.semilogy(xA * 2 * coordinates(xA_size), numpy.abs(error))
    ax2.semilogy(N * coordinates(xA_size), numpy.abs(fft(error)))
    err_sum += numpy.abs(error) ** 2 / nsubgrid
    err_sum_img += numpy.abs(fft(error)) ** 2 / nsubgrid
mark_range("$x_A$", -xA, xA, ax=ax1)
pylab.grid()
pylab.show()
# pylab.savefig("plot_error_facet_to_subgrid_1d.png")
mark_range("$N/2$", -N / 2, N / 2, ax=ax2)
pylab.grid()
pylab.show()
# pylab.savefig("plot_empty_n_per_2_1d.png")
print(
    "RMSE:",
    numpy.sqrt(numpy.mean(err_sum)),
    "(image:",
    numpy.sqrt(numpy.mean(err_sum_img)),
    ")",
)


#  By feeding the implementation single-pixel inputs we can create a full error map.

# ## Subgrid $\rightarrow$ facet

#
# We run into a very similar problem with $m$ as when reconstructing subgrids, except this time it happens because we want to construct:
# $$ b_j \left( m_i (n_j \ast S_i)\right)
#   = b_j \left( \mathcal F^{-1}\left[\Pi_{2y_P} \mathcal F m_i\right] (n_j \ast S_i)\right)$$
#
# As usual, this is entirely dual: In the previous case we had a signal limited by $y_B$ and needed the result of the convolution up to $y_N$, whereas now we have a signal bounded by $y_N$, but need the convolution result up to $y_B$. This cancels out - therefore we are okay with the same choice of $y_P$.

print("Subgrid data:", subgrid.shape, subgrid.size)
nafs = subgrid_to_facet_1d(
    subgrid, nsubgrid, nfacet, xM_yN_size, xM_size, facet_off, N, Fn
)

# - redistribution of FNjSi here -
print("Intermediate data:", nafs.shape, nafs.size)
approx_facet = reconstruct_facet_1d(
    nafs,
    nfacet,
    yB_size,
    nsubgrid,
    xMxN_yP_size,
    xM_yP_size,
    xN_yP_size,
    facet_m0_trunc,
    yP_size,
    subgrid_off,
    N,
    Fb,
    facet_B,
)
print("Reconstructed facets:", approx_facet.shape, approx_facet.size)


fig = pylab.figure(figsize=(16, 8))
ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
err_sum = err_sum_img = 0
for j in range(nfacet):
    error = approx_facet[j] - facet[j]
    err_sum += numpy.abs(ifft(error)) ** 2
    err_sum_img += numpy.abs(error) ** 2
    ax1.semilogy(coordinates(yB_size), numpy.abs(ifft(error)))
    ax2.semilogy(yB_size * coordinates(yB_size), numpy.abs(error))
print(
    "RMSE:",
    numpy.sqrt(numpy.mean(err_sum)),
    "(image:",
    numpy.sqrt(numpy.mean(err_sum_img)),
    ")",
)
mark_range("$x_A$", -xA, xA, ax=ax1)
mark_range("$x_M$", -xM, xM, ax=ax1)
mark_range("$y_B$", -yB, yB, ax=ax2)
mark_range("$0.5$", -0.5, 0.5, ax=ax1)

pylab.show()
# pylab.savefig("plot_error_subgrid_to_facet_1d.png")

# @interact_manual
# def generate_error_map_2():
#     error_map_2 = []
#     for xs in range(-N // 2, N // 2):
#         if xs % 128 == 0:
#             print(xs, end=" ")
#         FG = numpy.zeros(N)
#         FG[xs + N // 2] = 1
#         subgrid, facet = make_subgrid_and_facet(ifft(FG))
#         nafs = subgrid_to_facet_1(subgrid)
#
#         err_sum_hq = numpy.zeros(N, dtype=complex)
#
#         for j in range(nfacet):
#             approx = subgrid_to_facet_2(nafs, j)
#             err_sum_hq += numpy.roll(pad_mid(approx - facet[j], N), facet_off[j])
#         error_map_2.append(err_sum_hq)
#
#     err_abs = numpy.abs(error_map_2)
#     # Filter out spurious zeroes that would cause division-by-zero
#     err_log = numpy.log(
#         numpy.maximum(numpy.min(err_abs[err_abs > 0]), err_abs)
#     ) / numpy.log(10)
#     pylab.figure(figsize=(20, 20))
#     pylab.imshow(
#         err_log,
#         cmap=pylab.get_cmap("inferno"),
#         norm=colors.PowerNorm(gamma=2.0),
#         extent=(-N // 2, N // 2, -N // 2, N // 2),
#     )
#     pylab.colorbar(shrink=0.6)
#     pylab.ylabel("in")
#     pylab.xlabel("out")
#     pylab.title("Output error depending on input pixel (absolute log10)")
#     tikzplotlib.save(
#         "error_map_2.tikz",
#         axis_height="3.5cm",
#         axis_width="\\textwidth",
#         textsize=5,
#         show_info=False,
#     )
#     pylab.show()
#
#     worst_rmse = 0
#     worst_err = 0
#     for xs in range(N):
#         rmse = numpy.sqrt(numpy.mean(numpy.abs(error_map_2[xs]) ** 2))
#         if rmse > worst_rmse:
#             worst_rmse = rmse
#             worst_err = error_map_2[xs]
#     pylab.semilogy(numpy.abs(worst_err))


# ## 2D case
#
# All of this generalises to two dimensions in the way you would expect. Let us set up test data:


print(nsubgrid, "x", nsubgrid, "subgrids,", nfacet, "x", nfacet, "facets")
subgrid_2 = numpy.empty((nsubgrid, nsubgrid, xA_size, xA_size), dtype=complex)
facet_2 = numpy.empty((nfacet, nfacet, yB_size, yB_size), dtype=complex)

# adding sources
add_sources = True
if add_sources:
    FG_2 = numpy.zeros((N, N))
    source_count = 1000
    sources = [
        (
            numpy.random.randint(-N // 2, N // 2 - 1),
            numpy.random.randint(-N // 2, N // 2 - 1),
            numpy.random.rand() * N * N / numpy.sqrt(source_count) / 2,
        )
        for _ in range(source_count)
    ]
    for x, y, i in sources:
        FG_2[y + N // 2, x + N // 2] += i
    G_2 = ifft(FG_2)

else:
    # without sources
    G_2 = (
        numpy.exp(2j * numpy.pi * numpy.random.rand(N, N)) * numpy.random.rand(N, N) / 2
    )
    FG_2 = fft(G_2)

print("Mean grid absolute: ", numpy.mean(numpy.abs(G_2)))

for i0, i1 in itertools.product(range(nsubgrid), range(nsubgrid)):
    subgrid_2[i0, i1] = extract_mid(
        numpy.roll(G_2, (-subgrid_off[i0], -subgrid_off[i1]), (0, 1)), xA_size
    )
    subgrid_2[i0, i1] *= numpy.outer(subgrid_A[i0], subgrid_A[i1])
for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
    facet_2[j0, j1] = extract_mid(
        numpy.roll(FG_2, (-facet_off[j0], -facet_off[j1]), (0, 1)), yB_size
    )
    facet_2[j0, j1] *= numpy.outer(facet_B[j0], facet_B[j1])

# facet to subgrid test.
# Having those operations separately means that we can shuffle things around quite a bit
# without affecting the result. The obvious first choice might be to do all facet-preparation
# up-front, as this allows us to share the computation across all subgrids:
t = time.time()
NMBF_NMBF = numpy.empty(
    (nsubgrid, nsubgrid, nfacet, nfacet, xM_yN_size, xM_yN_size), dtype=complex
)
for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
    BF_F = prepare_facet(facet_2[j0, j1], 0, Fb, yP_size)
    BF_BF = prepare_facet(BF_F, 1, Fb, yP_size)
    for i0 in range(nsubgrid):
        NMBF_BF = extract_subgrid(
            BF_BF,
            i0,
            0,
            subgrid_off,
            yP_size,
            xMxN_yP_size,
            facet_m0_trunc,
            xM_yP_size,
            Fn,
            xM_yN_size,
            N,
        )
        for i1 in range(nsubgrid):
            NMBF_NMBF[i0, i1, j0, j1] = extract_subgrid(
                NMBF_BF,
                i1,
                1,
                subgrid_off,
                yP_size,
                xMxN_yP_size,
                facet_m0_trunc,
                xM_yP_size,
                Fn,
                xM_yN_size,
                N,
            )
print(time.time() - t, "s")

# # However, remember that `prepare_facet` increases the amount of data involved, which in turn
# means that we need to shuffle more data through subsequent computations.
# #
# # Therefore it is actually more efficient to first do the subgrid-specific reduction, and *then*
# continue with the (constant) facet preparation along the other axis. We can tackle both axes in
# whatever order we like, it doesn't make a difference for the result:
t = time.time()
for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
    BF_F = prepare_facet(facet_2[j0, j1], 0, Fb, yP_size)
    for i0 in range(nsubgrid):
        NMBF_F = extract_subgrid(
            BF_F,
            i0,
            0,
            subgrid_off,
            yP_size,
            xMxN_yP_size,
            facet_m0_trunc,
            xM_yP_size,
            Fn,
            xM_yN_size,
            N,
        )
        NMBF_BF = prepare_facet(NMBF_F, 1, Fb, yP_size)
        for i1 in range(nsubgrid):
            NMBF_NMBF[i0, i1, j0, j1] = extract_subgrid(
                NMBF_BF,
                i1,
                1,
                subgrid_off,
                yP_size,
                xMxN_yP_size,
                facet_m0_trunc,
                xM_yP_size,
                Fn,
                xM_yN_size,
                N,
            )
print(time.time() - t, "s")


t = time.time()
for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
    F_BF = prepare_facet(facet_2[j0, j1], 1, Fb, yP_size)
    for i1 in range(nsubgrid):
        F_NMBF = extract_subgrid(
            F_BF,
            i1,
            1,
            subgrid_off,
            yP_size,
            xMxN_yP_size,
            facet_m0_trunc,
            xM_yP_size,
            Fn,
            xM_yN_size,
            N,
        )
        BF_NMBF = prepare_facet(F_NMBF, 0, Fb, yP_size)
        for i0 in range(nsubgrid):
            NMBF_NMBF[i0, i1, j0, j1] = extract_subgrid(
                BF_NMBF,
                i0,
                0,
                subgrid_off,
                yP_size,
                xMxN_yP_size,
                facet_m0_trunc,
                xM_yP_size,
                Fn,
                xM_yN_size,
                N,
            )
print(time.time() - t, "s")


pylab.rcParams["figure.figsize"] = 16, 8
err_mean = err_mean_img = 0
for i0, i1 in itertools.product(range(nsubgrid), range(nsubgrid)):
    approx = numpy.zeros((xM_size, xM_size), dtype=complex)
    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        approx += numpy.roll(
            pad_mid(NMBF_NMBF[i0, i1, j0, j1], xM_size),
            (facet_off[j0] * xM_size // N, facet_off[j1] * xM_size // N),
            (0, 1),
        )
    approx = extract_mid(ifft(approx), xA_size)
    approx *= numpy.outer(subgrid_A[i0], subgrid_A[i1])
    err_mean += numpy.abs(approx - subgrid_2[i0, i1]) ** 2 / nsubgrid**2
    err_mean_img += numpy.abs(fft(approx - subgrid_2[i0, i1])) ** 2 / nsubgrid**2
pylab.imshow(numpy.log(numpy.sqrt(err_mean)) / numpy.log(10))
pylab.colorbar()
pylab.show()
# pylab.savefig("plot_error_mean_facet_to_subgrid_2d.png")
pylab.imshow(numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10))
pylab.colorbar()
pylab.show()
# pylab.savefig("plot_error_mean_image_facet_to_subgrid_2d.png")
print(
    "RMSE:",
    numpy.sqrt(numpy.mean(err_mean)),
    "(image:",
    numpy.sqrt(numpy.mean(err_mean_img)),
    ")",
)

test_accuracy_facet_to_subgrid(
    nsubgrid,
    xA_size,
    nfacet,
    yB_size,
    N,
    subgrid_off,
    subgrid_A,
    facet_off,
    facet_B,
    xM_yN_size,
    xM_size,
    Fb,
    yP_size,
    xMxN_yP_size,
    facet_m0_trunc,
    xM_yP_size,
    Fn,
    xs=252,
    ys=252,
)

### 2D case subgrid to facet
### This is based on the original implementation by Peter, and has not involved data redistribution yet.

# Verify that this is consistent with the previous implementation
nafs = subgrid_to_facet_1d(
    subgrid, nsubgrid, nfacet, xM_yN_size, xM_size, facet_off, N, Fn
)
for i in range(nsubgrid):
    FSi = prepare_subgrid(subgrid[i], xM_size)
    for j in range(nfacet):
        naf_new = extract_facet_contribution(
            FSi, Fn, facet_off, j, xM_size, N, xM_yN_size, 0
        )
        assert numpy.sqrt(numpy.average(numpy.abs(nafs[i, j] - naf_new) ** 2)) < 1e-14

approx_facet = numpy.array(
    [
        reconstruct_facet_1d(
            nafs,
            j,
            yB_size,
            nsubgrid,
            xMxN_yP_size,
            xM_yP_size,
            xN_yP_size,
            facet_m0_trunc,
            yP_size,
            subgrid_off,
            N,
            Fb,
            facet_B,
        )
        for j in range(nfacet)
    ]
)

for j in range(nfacet):
    MiNjSi_sum = numpy.zeros(yP_size, dtype=complex)
    for i in range(nsubgrid):
        add_subgrid_contribution(
            MiNjSi_sum,
            nafs[i, j],
            i,
            facet_m0_trunc,
            subgrid_off,
            xMxN_yP_size,
            xM_yP_size,
            yP_size,
            N,
            0,
        )
    approx_facet_new = finish_facet(MiNjSi_sum, Fb, facet_B, yB_size, j, 0)
    assert (
        numpy.sqrt(numpy.average(numpy.abs(approx_facet_new - approx_facet[j]) ** 2))
        < 1e-12
    )

# The actual calculation
t = time.time()
NAF_NAF = numpy.empty(
    (nsubgrid, nsubgrid, nfacet, nfacet, xM_yN_size, xM_yN_size), dtype=complex
)
for i0, i1 in itertools.product(range(nsubgrid), range(nsubgrid)):
    AF_AF = prepare_subgrid(subgrid_2[i0, i1], xM_size)
    for j0 in range(nfacet):
        NAF_AF = extract_facet_contribution(
            AF_AF, Fn, facet_off, j0, xM_size, N, xM_yN_size, 0
        )
        for j1 in range(nfacet):
            NAF_NAF[i0, i1, j0, j1] = extract_facet_contribution(
                NAF_AF, Fn, facet_off, j1, xM_size, N, xM_yN_size, 1
            )

BMNAF_BMNAF = numpy.empty((nfacet, nfacet, yB_size, yB_size), dtype=complex)
for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
    MNAF_BMNAF = numpy.zeros((yP_size, yB_size), dtype=complex)
    for i0 in range(nsubgrid):
        NAF_MNAF = numpy.zeros((xM_yN_size, yP_size), dtype=complex)
        for i1 in range(nsubgrid):
            add_subgrid_contribution(
                NAF_MNAF,
                NAF_NAF[i0, i1, j0, j1],
                i1,
                facet_m0_trunc,
                subgrid_off,
                xMxN_yP_size,
                xM_yP_size,
                yP_size,
                N,
                1,
            )
        NAF_BMNAF = finish_facet(NAF_MNAF, Fb, facet_B, yB_size, j1, 1)
        add_subgrid_contribution(
            MNAF_BMNAF,
            NAF_BMNAF,
            i0,
            facet_m0_trunc,
            subgrid_off,
            xMxN_yP_size,
            xM_yP_size,
            yP_size,
            N,
            0,
        )
    BMNAF_BMNAF[j0, j1] = finish_facet(MNAF_BMNAF, Fb, facet_B, yB_size, j0, 0)
print(time.time() - t, "s")

pylab.rcParams["figure.figsize"] = 16, 8
err_mean = err_mean_img = 0
for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
    approx = numpy.zeros((yB_size, yB_size), dtype=complex)
    approx += BMNAF_BMNAF[j0, j1]

    err_mean += numpy.abs(ifft(approx - facet_2[j0, j1])) ** 2 / nfacet**2
    err_mean_img += numpy.abs(approx - facet_2[j0, j1]) ** 2 / nfacet**2

pylab.imshow(numpy.log(numpy.sqrt(err_mean)) / numpy.log(10))
pylab.colorbar()
pylab.show()
# pylab.savefig("plot_error_mean_subgrid_to_facet_2d.png")
pylab.imshow(numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10))
pylab.colorbar()
pylab.show()
# pylab.savefig("plot_error_mean_image_subgrid_to_facet_2d.png")
print(
    "RMSE:",
    numpy.sqrt(numpy.mean(err_mean)),
    "(image:",
    numpy.sqrt(numpy.mean(err_mean_img)),
    ")",
)

test_accuracy_subgrid_to_facet(
    nsubgrid,
    xA_size,
    nfacet,
    yB_size,
    N,
    subgrid_off,
    subgrid_A,
    facet_off,
    facet_B,
    xM_yN_size,
    xM_size,
    Fb,
    yP_size,
    xMxN_yP_size,
    facet_m0_trunc,
    xM_yP_size,
    Fn,
    xs=252,
    ys=252,
)
