"""Condições de contorno dos cinco problemas elípticos da tese (Aplicação 1).

Faces: 'baixo' (y=0), 'cima' (y=L), 'esq' (x=0), 'dir' (x=L).
('D', valor) = Dirichlet; ('N', 0.0) = fluxo nulo.
"""
PROBLEMS = {
    "1.1": {"baixo": ("N", 0.0), "cima": ("N", 0.0), "esq": ("D", 1.0), "dir": ("D", 0.0)},
    "1.2": {"baixo": ("D", 1.0), "cima": ("D", 0.0), "esq": ("N", 0.0), "dir": ("N", 0.0)},
    "1.3": {"baixo": ("D", 1.0), "cima": ("N", 0.0), "esq": ("D", 0.0), "dir": ("N", 0.0)},
    "1.4": {"baixo": ("D", 0.0), "cima": ("D", 0.0), "esq": ("D", 1.0), "dir": ("D", 0.0)},
    "1.5": {"baixo": ("D", 1.0), "cima": ("D", 0.0), "esq": ("D", 1.0), "dir": ("D", 0.0)},
}

# Experimentos canônicos do upscaling (gradiente unitário em x e em y)
BC_UPSCALE_X = {"baixo": ("N", 0.0), "cima": ("N", 0.0), "esq": ("D", 1.0), "dir": ("D", 0.0)}
BC_UPSCALE_Y = {"baixo": ("D", 1.0), "cima": ("D", 0.0), "esq": ("N", 0.0), "dir": ("N", 0.0)}
