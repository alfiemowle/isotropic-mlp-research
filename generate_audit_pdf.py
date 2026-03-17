"""
Generate critical audit PDF using reportlab.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os

OUTPUT = os.path.join(os.path.dirname(__file__),
                      'dynamic_topology_net', 'results', 'CRITICAL_AUDIT.pdf')

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def style(name, **kw):
    s = ParagraphStyle(name, parent=styles['Normal'], **kw)
    return s

title_style   = style('Title',   fontSize=20, spaceAfter=6,  alignment=TA_CENTER,
                       textColor=colors.HexColor('#1a1a2e'), fontName='Helvetica-Bold')
subtitle_style= style('Sub',     fontSize=12, spaceAfter=18, alignment=TA_CENTER,
                       textColor=colors.HexColor('#444444'))
h1_style      = style('H1',      fontSize=15, spaceBefore=18, spaceAfter=8,
                       fontName='Helvetica-Bold',
                       textColor=colors.HexColor('#1a1a2e'))
h2_style      = style('H2',      fontSize=12, spaceBefore=12, spaceAfter=4,
                       fontName='Helvetica-Bold',
                       textColor=colors.HexColor('#c0392b'))
h3_style      = style('H3',      fontSize=11, spaceBefore=8,  spaceAfter=3,
                       fontName='Helvetica-Bold',
                       textColor=colors.HexColor('#2c3e50'))
body_style    = style('Body',    fontSize=9.5, spaceAfter=5,  leading=14,
                       alignment=TA_JUSTIFY)
mono_style    = style('Mono',    fontSize=8.5, spaceAfter=4,  leading=12,
                       fontName='Courier',
                       backColor=colors.HexColor('#f5f5f5'),
                       leftIndent=12, rightIndent=12)
bullet_style  = style('Bullet',  fontSize=9.5, spaceAfter=3,  leading=13,
                       leftIndent=18, bulletIndent=8)
warn_style    = style('Warn',    fontSize=9.5, spaceAfter=4,  leading=13,
                       backColor=colors.HexColor('#fff3cd'),
                       borderColor=colors.HexColor('#ffc107'),
                       borderWidth=1, borderPadding=6,
                       leftIndent=8, rightIndent=8)
severe_style  = style('Severe',  fontSize=9.5, spaceAfter=4,  leading=13,
                       backColor=colors.HexColor('#f8d7da'),
                       borderColor=colors.HexColor('#dc3545'),
                       borderWidth=1, borderPadding=6,
                       leftIndent=8, rightIndent=8)
good_style    = style('Good',    fontSize=9.5, spaceAfter=4,  leading=13,
                       backColor=colors.HexColor('#d4edda'),
                       borderColor=colors.HexColor('#28a745'),
                       borderWidth=1, borderPadding=6,
                       leftIndent=8, rightIndent=8)

def P(text, s=body_style):
    return Paragraph(text, s)

def B(text):
    return Paragraph(f'&#8226; {text}', bullet_style)

def HR():
    return HRFlowable(width='100%', thickness=0.5,
                      color=colors.HexColor('#cccccc'), spaceAfter=6)

def severity_badge(level):
    colours = {'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'}
    c = colours.get(level, '#999999')
    return f'<font color="{c}"><b>[{level}]</b></font>'


# ── Issues ────────────────────────────────────────────────────────────────────

issues = [

    # ── 1 ──────────────────────────────────────────────────────────────────────
    {
        'id': 1,
        'title': 'Single seed (42) across nearly all experiments',
        'severity': 'HIGH',
        'affects': 'All quantitative conclusions — virtually every test',
        'problem': (
            'Of the 38 experiments, the vast majority use only seed=42. '
            'A single random initialisation cannot distinguish a real effect from a lucky draw. '
            'Even tests with seed=[42,123] (E, M, Q, U2, W, Y, Z) use only two seeds. '
            'With neural networks on CIFAR-10, run-to-run variance is typically 0.5-1.5pp, '
            'meaning claimed differences of <1pp (e.g. AN: +0.0024 for diag vs static, '
            'AP: +0.0143 for pruning) may be entirely within noise. '
            'The highest-stakes finding -- Iso-first-4L at 52.28% being best -- rests '
            'on a single seed at a single width.'
        ),
        'why_matters': (
            'Every numerical claim in the study could be a seed artefact. '
            'Conclusions about "which architecture is best" or "Adam reset hurts" '
            'are not statistically defensible without variance estimates. '
            'A reviewer would immediately reject any paper built on this evidence base.'
        ),
        'proposed_test': (
            'Priority replication: re-run the 10 most important experiments '
            '(AN, AP, AQ, AR, AS, AE, AI, J, U2, Z) with seeds [42, 123, 7, 0, 99]. '
            'Report mean +/- std. Any claimed difference smaller than 2x std should '
            'be labelled inconclusive. This is the single highest-value experiment '
            'that could be run — one seed replication run changes the evidential '
            'weight of every conclusion in the project.'
        ),
    },

    # ── 2 ──────────────────────────────────────────────────────────────────────
    {
        'id': 2,
        'title': 'Baseline is elementwise tanh without normalisation -- a strawman',
        'severity': 'HIGH',
        'affects': 'Tests D, E, L, M, Q, R, X, AG, AE, AD -- the core accuracy advantage claim',
        'problem': (
            'The "Base" model used throughout the early experiments is an MLP with '
            'elementwise tanh and no LayerNorm, no BatchNorm, no residual connections. '
            'This is not a representative modern baseline. By 2020, no practitioner '
            'would deploy such a model. Test AE eventually added LN+tanh and found it '
            'beats Iso at 24 epochs (+3pp). Test AH found LN+GELU and LN+SiLU '
            'both beat Iso. The claimed "+14-16% on CIFAR-10" advantage for Iso '
            'is almost entirely a comparison against this weak strawman, not against '
            'state-of-the-art normalised networks.'
        ),
        'why_matters': (
            'The paper\'s main empirical claim -- Iso outperforms standard architectures -- '
            'is primarily validated against Base-tanh. When compared fairly to LN+GELU '
            '(the standard modern FFN used in BERT, GPT, etc.), Iso loses (47.6% vs 44.7% '
            'at depth 3). The entire framing of "Iso is a better activation" is undermined. '
            'The correct framing is: "Iso enables dynamic topology; LN+GELU is better for '
            'raw accuracy; hybrids may get both." This is a weaker but more defensible claim.'
        ),
        'proposed_test': (
            'Drop "Base" from all comparative claims. Replace with: '
            '(1) LN+GELU as the standard baseline, '
            '(2) Iso vs LN+GELU at matched training budget and width, '
            '(3) Iso-first hybrid as the practical recommendation. '
            'Report the absolute gap between Iso and LN+GELU, not Iso and Base-tanh.'
        ),
    },

    # ── 3 ──────────────────────────────────────────────────────────────────────
    {
        'id': 3,
        'title': 'CPU vs GPU inconsistency across experiments',
        'severity': 'MEDIUM',
        'affects': 'AG (CUDA), AG-B (CPU), AK-AM (CPU), AN-AR (CPU), all earlier tests (mixed)',
        'problem': (
            'Results were produced on different hardware across the suite. '
            'AG ran on CUDA; AG-B ran on CPU (after a sequencing error forced it). '
            'AK-AR all ran on CPU. Early experiments (A-Z) ran on whatever was available. '
            'While gradient descent results should be device-independent given the same '
            'floating-point precision, in practice CUDA and CPU may give different results '
            'due to: different order of floating-point operations (non-associativity), '
            'different default data types in some operations, and potential CUDA-specific '
            'cuDNN kernel choices that are non-deterministic by default. '
            'This means AG (CUDA) and AG-B (CPU) are not directly comparable -- '
            'the conclusion that "LN+tanh beats Iso at depth 4" rests on cross-device comparison.'
        ),
        'why_matters': (
            'The AG vs AG-B comparison is used to support several claims about depth scaling. '
            'If CUDA and CPU give different numerical results (which is possible even with '
            'the same model and seed), the cross-device delta is confounded. '
            'At the scale of differences we are discussing (1-3pp), device effects could matter.'
        ),
        'proposed_test': (
            'Re-run both AG and AG-B on the same device (preferably CUDA for speed). '
            'Also set torch.backends.cudnn.deterministic=True and '
            'torch.backends.cudnn.benchmark=False when using CUDA to ensure reproducibility. '
            'Add torch.use_deterministic_algorithms(True) where possible.'
        ),
    },

    # ── 4 ──────────────────────────────────────────────────────────────────────
    {
        'id': 4,
        'title': 'Test J (fair comparison) used only 72 epochs -- too short',
        'severity': 'MEDIUM',
        'affects': 'Test J -- the key "dynamic topology does not add accuracy" conclusion',
        'problem': (
            'Test J compared Static-Iso and Dynamic-Iso both at 72 epochs and found them '
            'essentially equal (40.3% vs 40.2%). This was used to conclude that '
            '"dynamic topology provides flexibility, not free accuracy." '
            'However, AI showed that Iso keeps improving slowly past epoch 100, '
            'while LN+tanh overfits after epoch 25. If dynamic topology acts as '
            'implicit regularisation (AP showed pruning can improve accuracy by +1.4pp), '
            'then the benefit might only appear at longer training horizons. '
            'Test J was never extended to 100+ epochs.'
        ),
        'why_matters': (
            'If dynamic topology provides regularisation that prevents overfitting at long '
            'training, the conclusion of Test J is wrong at the timescales that matter. '
            'Test AS is running 100-epoch dynamic vs static, but uses lr=0.001 '
            'and a different architecture than Test J. A direct comparison is needed.'
        ),
        'proposed_test': (
            'Extend Test J to 150 epochs with the same architecture (IsotropicMLP 1L w=32). '
            'Compare: (1) Static-Iso 150ep, (2) Dynamic-Iso 150ep (diag every 5ep, prune 32->24 at ep60). '
            'Track both accuracy and overfitting indicators (train vs test gap). '
            'If dynamic wins by >1pp at ep150 with variance across 3+ seeds, the '
            'regularisation hypothesis is confirmed.'
        ),
    },

    # ── 5 ──────────────────────────────────────────────────────────────────────
    {
        'id': 5,
        'title': 'Test AM multi-layer pruning bug affects conclusions',
        'severity': 'MEDIUM',
        'affects': 'Test AM -- the "LN+tanh topology is viable" conclusion',
        'problem': (
            'Test AM used W1 row norms as a pruning proxy for a multi-layer network, '
            'applying the SAME keep indices to all layers. This is incorrect: '
            'each layer boundary should be independently evaluated. '
            'Test AP fixed this for Iso networks. However, AM\'s LN+tanh pruning '
            'results were not reproduced with the corrected method. '
            'The "LN+tanh recovers 92% in 1 epoch" conclusion uses the buggy pruning. '
            'The actual recovery rate under correct W2-norm pruning (identified in AJ '
            'as the right criterion for LN+tanh) is unknown.'
        ),
        'why_matters': (
            'The conclusion that LN+tanh topology is "viable" rests on pruning results '
            'from a buggy implementation. If correct pruning gives better or worse recovery, '
            'the viability assessment changes. This is particularly important because AM\'s '
            'comparison (Iso exact vs LN+tanh approximate) is used to position the '
            'practical value of the paper\'s approach.'
        ),
        'proposed_test': (
            'Re-run AM\'s pruning conditions using: '
            '(1) AJ\'s W2-norm criterion for LN+tanh, and '
            '(2) AP\'s proper chained SVD for Iso (as a cross-check). '
            'Compare recovery curves (0, 1, 2, 5 fine-tune epochs) for both. '
            'This directly validates whether the AM conclusions hold under correct methodology.'
        ),
    },

    # ── 6 ──────────────────────────────────────────────────────────────────────
    {
        'id': 6,
        'title': 'CIFAR-10 as flattened vectors ignores spatial structure',
        'severity': 'MEDIUM',
        'affects': 'All CIFAR-10 experiments -- the primary benchmark',
        'problem': (
            'All experiments use CIFAR-10 images flattened to 3072-dimensional vectors '
            'fed into an MLP. This discards all spatial structure (locality, translation '
            'invariance) that makes CIFAR-10 hard. A flat MLP on CIFAR-10 is limited to '
            '~55-60% accuracy regardless of activation choice; CNNs achieve 90%+. '
            'At the accuracy range we are working in (40-52%), the differences are '
            'dominated by how well the activation handles the statistical structure '
            'of flattened natural images -- not by general-purpose learning ability. '
            'The O(n)-equivariance property of Iso may be accidentally well-suited '
            'to this specific task structure in ways that would not generalise.'
        ),
        'why_matters': (
            'Claims like "Iso beats LN+GELU" or "hybrids achieve 52.28%" may be '
            'artefacts of the flat-CIFAR-10 setting. On a structured task with a '
            'proper inductive bias (e.g. sequence modelling, graph classification, '
            'or even MNIST/F-MNIST at higher width), the relative rankings might differ. '
            'L (cross-dataset) tested MNIST/F-MNIST but at width=24 and 24 epochs -- '
            'a very limited test.'
        ),
        'proposed_test': (
            'Test at least one structured task: '
            '(a) Sequence classification (e.g. permuted MNIST, which tests representation '
            'quality under a fixed order), or '
            '(b) A regression task where gradient flow quality directly determines accuracy. '
            'The comparison should focus on depth scaling (does Iso scale better with depth '
            'on a structured task?), not raw accuracy. This would validate the gradient '
            'flow mechanism claim beyond the CIFAR-10 setting.'
        ),
    },

    # ── 7 ──────────────────────────────────────────────────────────────────────
    {
        'id': 7,
        'title': 'No learning rate scheduling -- all experiments use constant LR Adam',
        'severity': 'MEDIUM',
        'affects': 'AI, AF, AR, AS -- long-training and width-scaling results',
        'problem': (
            'Every experiment uses constant learning rate Adam throughout training. '
            'AQ showed that lr=0.001 is better than lr=0.08 for Iso-family models, '
            'but a cosine decay schedule (standard in modern training) would likely '
            'outperform both. In AI, the observation that "LN+tanh peaks at ep25 then '
            'degrades" is almost certainly an overfitting symptom of constant high LR -- '
            'a warmup+cosine schedule would likely prevent this degradation entirely, '
            'invalidating the "Iso overtakes LN+tanh at ep100" conclusion. '
            'AS runs 100 epochs at lr=0.001 constant -- results will be suboptimal '
            'for both conditions (LN and Iso).'
        ),
        'why_matters': (
            'The long-training conclusions (Iso more stable than LN, Iso overtakes LN '
            'at ep100) rest on the specific behaviour of constant-LR Adam. With a proper '
            'schedule, LN+tanh would not overfit, and the "Iso catches up" narrative '
            'might disappear. This makes AI\'s key finding fragile.'
        ),
        'proposed_test': (
            'Re-run AI with cosine LR decay (warmup 5 epochs, decay to 1e-5 by ep100) '
            'for Iso, LN+tanh, LN+Iso, RMS+tanh. If LN+tanh stops degrading after ep25, '
            'the "Iso overtakes" conclusion is an LR schedule artefact. '
            'This is a single run that could change the AI conclusion entirely.'
        ),
    },

    # ── 8 ──────────────────────────────────────────────────────────────────────
    {
        'id': 8,
        'title': 'Test AK IsoGELU conclusions drawn at wrong LR -- partially invalidated',
        'severity': 'MEDIUM',
        'affects': 'Test AK conclusions, and any downstream claims based on them',
        'problem': (
            'Test AK concluded IsoGELU/SiLU are "dramatically worse" and '
            '"the hypothesis that non-saturating sigma extends the depth ceiling was wrong." '
            'Test AQ then showed this was entirely an LR artefact: at lr=0.001, '
            'the gap collapses from ~12pp to ~1pp. The AK conclusion was too strong. '
            'Furthermore, AQ only tested depth=3 at low LR. The original AK claim '
            'was specifically about extending the depth ceiling beyond 4L. '
            'This was never tested at low LR -- IsoGELU at depths 1-5 with lr=0.001 '
            'has not been run. The depth ceiling claim remains unresolved.'
        ),
        'why_matters': (
            'If IsoGELU at low LR extends the depth ceiling to 5-6L (where IsoTanh '
            'collapses), this would be a genuinely new finding: non-saturating sigma '
            'with appropriate LR enables deeper Iso networks. The conclusion recorded '
            'in ALL_RESULTS.md ("IsoGELU is viable with proper LR tuning") is accurate '
            'but incomplete -- the depth extension hypothesis remains open.'
        ),
        'proposed_test': (
            'Re-run AK at lr=0.001: IsoTanh/IsoGELU/IsoSiLU at depths 1-6, '
            'width=32, 60 epochs (longer to see if depth>4 stabilises). '
            'Specifically test depths 5 and 6 where IsoTanh fails. '
            'If IsoGELU at depth=5/6 with lr=0.001 achieves >40%, the depth '
            'ceiling extension hypothesis is confirmed.'
        ),
    },

    # ── 9 ──────────────────────────────────────────────────────────────────────
    {
        'id': 9,
        'title': 'Composite pruning criterion never tested under interleaved protocol',
        'severity': 'MEDIUM',
        'affects': 'Test AS (current run) and the pruning criterion recommendations',
        'problem': (
            'The composite criterion (Sigma_ii x ||W2-col||) was validated in U2 and Z '
            'as the best criterion in a static, post-hoc pruning context: train to convergence, '
            'diagonalise once, evaluate per-neuron accuracy drop, compare correlations. '
            'Test AS now uses this criterion in an interleaved setting where diagonalisation '
            'happens every 5 epochs and pruning happens before convergence. '
            'The criterion was never validated in this dynamic context. '
            'In particular, early in training (e.g. ep20), the SV spectrum may not be '
            'informative (condition number is ~4.0 at ep1, ~8.4 at ep24 per Test S), '
            'so early pruning decisions guided by the composite criterion may be suboptimal.'
        ),
        'why_matters': (
            'If the criterion degrades under interleaved use, AS may be pruning the '
            'wrong neurons and underestimating the performance of the full pipeline. '
            'The validation (U2, Z) and the application (AS) are in different regimes.'
        ),
        'proposed_test': (
            'In the AS run (or a follow-up), track per-pruning-event accuracy drop immediately '
            'after each prune (before fine-tune). Also track the SV spectrum condition number '
            'at each prune event. If early prune events (ep20, ep30) cause larger drops than '
            'late events, this suggests the criterion is less reliable early. '
            'Compare against random pruning at each event as a baseline -- '
            'if composite criterion is not clearly better than random during interleaved training, '
            'the dynamic criterion needs re-validation.'
        ),
    },

    # ── 10 ─────────────────────────────────────────────────────────────────────
    {
        'id': 10,
        'title': 'Test AF width scaling saturates but mechanism is untested',
        'severity': 'MEDIUM',
        'affects': 'Test AF -- width scaling conclusions',
        'problem': (
            'Test AF found Iso accuracy essentially flat at 42-43% from width=32 to w=512 '
            'at 24 epochs, while LN+tanh improves from 45% to 48%. '
            'The interpretation given was "Iso is near its capacity ceiling at 24 epochs." '
            'However, there are at least two alternative explanations: '
            '(a) Iso needs more epochs at larger width (the optimisation landscape is '
            'harder to navigate with the norm-dependent activation at large scale), or '
            '(b) The fixed LR=0.08 is too high at large width for Iso '
            '(AQ showed LR sensitivity; wider networks may need lower LR). '
            'These hypotheses were not tested. The "capacity ceiling" interpretation '
            'could be entirely wrong.'
        ),
        'why_matters': (
            'If Iso at w=512 with lr=0.001 and 100 epochs achieves significantly more '
            'than 42%, then the AF conclusion ("Iso does not scale with width") is false. '
            'This directly impacts recommendations about when to use Iso vs LN+GELU.'
        ),
        'proposed_test': (
            'Re-run AF with: (1) lr=0.001 at widths [32, 128, 512], 60 epochs, '
            'comparing Iso vs LN+GELU. If Iso improves with width at low LR, '
            'the ceiling was an LR artefact. '
            'If Iso is still flat at w=512 with low LR and 60 epochs, '
            'the capacity ceiling interpretation is supported.'
        ),
    },

    # ── 11 ─────────────────────────────────────────────────────────────────────
    {
        'id': 11,
        'title': 'Scaffold inertness tested only at float32 -- not robust to large biases',
        'severity': 'LOW',
        'affects': 'Tests A, B -- the core mathematical foundations',
        'problem': (
            'Tests A and B confirm reparameterisation invariance and neurogenesis invariance '
            'to float32 precision (max diff ~3e-5). These tests used the default trained model '
            'at a moderate training point. The intrinsic length test (C) showed that IL '
            'correction is negligible "when biases are small." But the tests did not explore: '
            '(a) What happens at large bias values (bias magnitudes >> activation norm)? '
            '(b) What happens after many diagonalise-prune cycles where accumulated '
            'floating-point errors compound? '
            '(c) Whether scaffold inertness holds at float16 (used in production)? '
            'Test AA showed IL can explode (seed 123, o -> 10^12) under certain conditions, '
            'suggesting numerical instability is a real risk not fully characterised.'
        ),
        'why_matters': (
            'The mathematical claims A and B are the bedrock of the paper. '
            'If they fail at large bias or after many operations, the dynamic topology '
            'mechanism is unreliable in practice, even if it works in the tested regime.'
        ),
        'proposed_test': (
            'Test A robustness: run 100 sequential diagonalise operations on the same '
            'model and track accumulated error. Test B robustness: test with deliberately '
            'large biases (b ~ 5x normal init scale) and with models trained for 200 epochs '
            'where biases have had time to grow. Test at float16 and bfloat16. '
            'Report max cumulative error vs number of operations.'
        ),
    },

    # ── 12 ─────────────────────────────────────────────────────────────────────
    {
        'id': 12,
        'title': 'Test J comparison: parameters are not matched between Dynamic and Static',
        'severity': 'MEDIUM',
        'affects': 'Test J -- the "dynamic topology does not add accuracy" conclusion',
        'problem': (
            'Test J compares Static-Iso (w=24) vs Dynamic-Iso (starts w=32, prunes to w=24). '
            'By the end of training both have width=24 and should have similar parameter counts, '
            'but during training the Dynamic model has MORE parameters (extra neurons '
            'contribute to the gradient landscape even while being pruned). '
            'The Dynamic model effectively gets to train a w=32 model for 48 epochs and '
            'then a w=24 model for 24 more -- this is richer training signal, not just '
            'equal-epoch comparison. '
            'A truly fair comparison would also include: Static-Iso w=32 for 72 epochs, '
            'which has the same total computation but no pruning. This baseline is missing.'
        ),
        'why_matters': (
            'If Static-Iso w=32 for 72 epochs outperforms Dynamic-Iso 72ep (which ends at w=24), '
            'the correct conclusion is that dynamic topology reduces final model quality '
            'vs training an oversized static model. If Dynamic-Iso beats Static-Iso w=32, '
            'then pruning does add value. The current test cannot distinguish these cases.'
        ),
        'proposed_test': (
            'Four-way comparison at 72 epochs: '
            '(1) Static-Iso w=24, (2) Static-Iso w=32, '
            '(3) Dynamic-Iso 32->24, (4) Static-Iso w=32 then fine-tune w=24 (manual prune at ep48). '
            'Seeds [42, 123, 7]. This isolates the value of dynamic topology '
            'from the value of starting at larger width.'
        ),
    },

    # ── 13 ─────────────────────────────────────────────────────────────────────
    {
        'id': 13,
        'title': 'Test AE/AI conflict with AH: LN+SiLU beats LN+tanh but is never used in hybrids',
        'severity': 'LOW',
        'affects': 'AL, AR, AS -- hybrid architecture design choices',
        'problem': (
            'Test AH found LN+SiLU (47.80%) and LN+GELU (47.62%) both outperform '
            'LN+tanh (46.54%) at depth=3, 24 epochs. Yet all hybrid architecture tests '
            '(AL, AR, AS) use Iso + LN+GELU as the non-Iso layers. '
            'The choice of LN+GELU over LN+SiLU was never justified. '
            'If LN+SiLU is the better LN-based activation, then Iso-first + LN+SiLU layers '
            'would be a stronger hybrid than Iso-first + LN+GELU. '
            'The best hybrid configuration was never systematically searched -- '
            'AL only tested the Iso position (first/last/sandwich), not the LN layer type.'
        ),
        'why_matters': (
            'The claim that "Iso-first-4L at 52.28% is the best result in the entire study" '
            'may underestimate what is achievable. Iso-first + LN+SiLU layers might achieve '
            '53-54%. If so, the "best configuration" finding is incomplete.'
        ),
        'proposed_test': (
            'Run AL-style comparison with LN+SiLU and LN+GELU at width=128: '
            'Pure-LN+GELU-4L, Pure-LN+SiLU-4L, Iso-first+LNG-4L, Iso-first+LNSiLU-4L. '
            '30 epochs. Determine if the Iso+LN+SiLU hybrid beats Iso+LN+GELU, '
            'and whether LN+SiLU as the body activation changes the optimal Iso position.'
        ),
    },

    # ── 14 ─────────────────────────────────────────────────────────────────────
    {
        'id': 14,
        'title': 'Test AN used only 1L model -- interleaved protocol not tested at depth',
        'severity': 'MEDIUM',
        'affects': 'Test AN -- the "stale Adam is safe" conclusion',
        'problem': (
            'Test AN used IsotropicMLP with a single hidden layer (1L). '
            'The conclusion that "stale Adam momentum after partial_diagonalise is not harmful" '
            'was drawn from this 1L model. However, the effect of stale Adam may be depth-dependent: '
            'at 1L, the reparameterisation only rotates W1 and affects b1, '
            'while W2 (to output) absorbs U. At 3L or 4L, the rotation propagates through '
            'multiple Adam state tensors simultaneously. The staleness effect may compound '
            'at depth in ways not captured by the 1L test. '
            'Test AS runs the interleaved protocol on Iso-first-4L, but only the '
            'Iso layer (1 of 4 layers) is diagonalised -- the 3 LN+GELU layers are untouched. '
            'A 4L pure-Iso model with interleaved diagonalise was never tested.'
        ),
        'why_matters': (
            'If stale Adam is harmful at depth, the AS run may be suppressed by this effect '
            'without us knowing. The AN result that "reset hurts" was from 1L and may not '
            'hold at depth where accumulated curvature estimates are less reliable. '
            'This could explain unexpected underperformance in AS.'
        ),
        'proposed_test': (
            'Replicate AN conditions at depth=3: '
            '(A) Static-Iso-3L, (B) Diag-only-3L (all layers diagonalised every 5ep), '
            '(C) Diag+reset-3L. Compare the A vs B and B vs C gaps at 3L vs 1L. '
            'If the reset helps at depth=3 but hurts at depth=1, the conclusion is depth-dependent '
            'and must be qualified.'
        ),
    },

    # ── 15 ─────────────────────────────────────────────────────────────────────
    {
        'id': 15,
        'title': 'Width=32 used in most tests is unrepresentatively small',
        'severity': 'MEDIUM',
        'affects': 'Tests A-Z (most), AK, AL -- any conclusion about relative model behaviour',
        'problem': (
            'The majority of experiments use width=32. This is a very narrow MLP by '
            'any modern standard -- typical hidden dims are 256-4096. At width=32 on '
            'CIFAR-10 (3072 input), the network is severely underparameterised: '
            'W1 is 32x3072 = 98K parameters, W2 is 32x32 = 1K. '
            'The dynamics of isotropic activation, SVD spectrum, and pruning at '
            'width=32 may be qualitatively different from realistic widths. '
            'For example, at w=32 there are only 32 singular values to differentiate; '
            'at w=512 the spectrum has room for much richer structure. '
            'Test AR showed that going from w=32 to w=128 already changed '
            'which hybrid position is best (Iso-last at w=32, Iso-first at w=128).'
        ),
        'why_matters': (
            'Small-width findings may not generalise. The entire pruning criterion analysis '
            '(U2, Z) was done at w=16 and w=24. At larger widths, the composite criterion '
            'may perform differently, the pruning recovery dynamics change, and the '
            'optimal timing may shift. Claims derived from w=32 experiments should be '
            'explicitly flagged as small-scale results pending replication at realistic widths.'
        ),
        'proposed_test': (
            'Priority scale-up test: re-run Tests G (pruning criterion), H (sequential '
            'pruning stability), and V (pruning timing) at width=128 and width=256. '
            'Compare r(SV), r(Composite) at larger widths. '
            'If correlations change significantly, the criterion recommendations need revision.'
        ),
    },

    # ── 16 ─────────────────────────────────────────────────────────────────────
    {
        'id': 16,
        'title': 'Test AP 2L pruning improvement (+1.4pp) has no mechanistic explanation',
        'severity': 'LOW',
        'affects': 'Test AP and the interpretation of pruning as regularisation',
        'problem': (
            'AP found that pruning a 2L Iso network from w=32 to w=24 and fine-tuning '
            '5 epochs increased accuracy by +1.4pp. This was attributed to regularisation '
            '("removing redundant neurons that act as noise"). However, this was observed '
            'on a single seed with a 5ep fine-tune window. Possible alternative explanations: '
            '(a) The pruned model is evaluated differently (fewer params = fewer overfit features), '
            '(b) 5 fine-tune epochs happen to land at a better point in the loss landscape '
            'by chance for this seed, '
            '(c) The accuracy of the unpruned baseline (39.85%) was suboptimally trained '
            '(lower than typical 2L at w=32 which is ~43-44%), suggesting the baseline '
            'model was particularly weak, making relative improvement easy.'
        ),
        'why_matters': (
            'If the +1.4pp finding is a seed artefact, then the "pruning acts as structural '
            'regularisation" narrative is unsupported. More importantly, the 2L-baseline '
            'accuracy of 39.85% is anomalously low (AP used bias=False for hidden layers, '
            'which likely explains this). The AP improvement may just be recovering from '
            'a suboptimal baseline rather than genuine regularisation.'
        ),
        'proposed_test': (
            'Re-run AP with bias=True in hidden layers (matching the architecture used '
            'in other experiments). Test 3 seeds. Measure whether the +1.4pp improvement '
            'persists. If the baseline accuracy rises to ~43% (matching other 2L results) '
            'but the post-prune gain shrinks, the improvement was an artefact of the '
            'no-bias architecture. Track the full fine-tune curve (ep0-20) not just ep5.'
        ),
    },

    # ── 17 ─────────────────────────────────────────────────────────────────────
    {
        'id': 17,
        'title': 'Test AB shell collapse width scaling: wrong direction of residual growth',
        'severity': 'LOW',
        'affects': 'Test AB -- the "genuine gap in Appendix C" conclusion',
        'problem': (
            'AB found that the affine residual under L2-normalised inputs GROWS with width '
            '(0.0051 at w=32 to 0.0155 at w=256). This was interpreted as evidence of '
            'a "genuine unresolved gap in Appendix C." However, there is a simpler '
            'explanation: the lstsq fit has higher estimation variance at larger width '
            'because the parameter space of the affine map grows as O(d^2) while the '
            'test set is fixed at 10K samples. A larger affine map is harder to fit '
            'from finite data, so the residual appears to grow even if the true '
            '(population-level) residual is zero. The test did not control for this '
            'statistical confound.'
        ),
        'why_matters': (
            'If the residual growth is a finite-sample estimation artefact, then '
            'Appendix C may actually be correct and shell collapse is exact in the '
            'population limit. The conclusion "genuine gap in Appendix C" would be wrong. '
            'This matters for the theoretical framing of the paper.'
        ),
        'proposed_test': (
            'Test AB with a LARGER test set at larger widths: use the full 50K training '
            'set as the lstsq fitting set and the 10K test set as the evaluation set. '
            'If the residual still grows with width under this controlled condition '
            '(where estimation variance is no longer confounded), then the gap is genuine. '
            'If it shrinks or stabilises, it was a finite-sample artefact.'
        ),
    },

    # ── 18 ─────────────────────────────────────────────────────────────────────
    {
        'id': 18,
        'title': 'Test R / Test X had measurement bugs -- conclusions rest on flawed data',
        'severity': 'MEDIUM',
        'affects': 'Tests R, X -- the mechanistic explanation of Base depth failure',
        'problem': (
            'Test R noted that "eff_rank tracking non-functional (NaN for all -- hook issue '
            'with custom activation modules)." The effective rank metric was supposed to '
            'be the key measurement, but it failed silently. Only gradient norms were '
            'successfully tracked. Test X also noted "eff_rank tracking non-functional" '
            'for the same reason. The conclusion that "Base depth failure is not '
            'representational collapse" (drawn from AC, which DID work correctly) '
            'is sound, but R\'s autopsy of the MECHANISM (gradient concentration in '
            'output layer) was derived from a run where one of the intended measurements failed. '
            'The gradient norm findings in R are likely correct, but they were not '
            'the primary intended measurement.'
        ),
        'why_matters': (
            'The mechanism explanation (gradient concentration in output layer) is '
            'one of the most interesting findings in the suite -- it explains WHY '
            'Base depth fails and WHY elementwise tanh saturation is the root cause. '
            'But the supporting measurement (eff_rank) never worked. The hook issue '
            'should be fixed and R/X re-run with all measurements working to provide '
            'complete mechanistic evidence.'
        ),
        'proposed_test': (
            'Fix the hook-based effective rank measurement (use a forward hook that '
            'captures the hidden representation tensor directly, rather than relying '
            'on a hook with custom activation modules). Re-run AD and X with working '
            'eff_rank tracking. Verify that the PR (participation ratio, which DID work '
            'in AC) agrees with the hook-based eff_rank across models.'
        ),
    },

]


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title='Critical Audit — Isotropic MLP Research',
        author='Claude Sonnet 4.6 (self-audit)',
    )

    story = []

    # Cover
    story += [
        Spacer(1, 1*cm),
        P('CRITICAL AUDIT', title_style),
        P('Isotropic MLP Research — Experimental Suite (Tests A–AS)', subtitle_style),
        P('Self-audit by Claude Sonnet 4.6 | 2026-03-17', subtitle_style),
        HR(),
        Spacer(1, 0.5*cm),
        P(
            'This document is a critical self-assessment of 38 experiments '
            'conducted to validate and extend the paper <i>"On De-Individuated Neurons: '
            'Continuous Symmetries Enable Dynamic Topologies"</i> (Bird, 2026). '
            'For each identified issue, the document states: what the problem is, '
            'why it matters for the credibility of the conclusions, and exactly '
            'how a targeted experiment would resolve it. '
            'The aim is to distinguish conclusions that are robust and defensible '
            'from those that require further validation before being stated with confidence.',
            body_style
        ),
        Spacer(1, 0.5*cm),
    ]

    # Severity summary table
    high   = [i for i in issues if i['severity'] == 'HIGH']
    medium = [i for i in issues if i['severity'] == 'MEDIUM']
    low    = [i for i in issues if i['severity'] == 'LOW']

    table_data = [
        ['Severity', 'Count', 'Issues'],
        ['HIGH',   str(len(high)),   ', '.join(f"#{i['id']}" for i in high)],
        ['MEDIUM', str(len(medium)), ', '.join(f"#{i['id']}" for i in medium)],
        ['LOW',    str(len(low)),    ', '.join(f"#{i['id']}" for i in low)],
    ]
    t = Table(table_data, colWidths=[3*cm, 2*cm, 11.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f8d7da')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#fff3cd')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#d4edda')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [None]),
    ]))
    story += [t, Spacer(1, 0.5*cm)]

    story.append(PageBreak())

    # Issues
    story.append(P('ISSUES', h1_style))
    story.append(HR())

    for issue in issues:
        sev   = issue['severity']
        badge = severity_badge(sev)
        sty   = severe_style if sev == 'HIGH' else (warn_style if sev == 'MEDIUM' else good_style)

        story += [
            Spacer(1, 0.3*cm),
            P(f'Issue #{issue["id"]} {badge}  —  {issue["title"]}', h2_style),
            P(f'<b>Affects:</b> {issue["affects"]}', body_style),
            Spacer(1, 0.15*cm),
            P('<b>What is the problem?</b>', h3_style),
            P(issue['problem'], sty),
            Spacer(1, 0.1*cm),
            P('<b>Why does it matter?</b>', h3_style),
            P(issue['why_matters'], body_style),
            Spacer(1, 0.1*cm),
            P('<b>Proposed test to resolve it</b>', h3_style),
            P(issue['proposed_test'], good_style),
            HR(),
        ]

    # Closing summary
    story += [
        PageBreak(),
        P('SUMMARY: WHICH CONCLUSIONS ARE ROBUST?', h1_style),
        HR(),
        Spacer(1, 0.3*cm),
        P('<b>Highly robust (multiple seeds, clean methodology, confirmed by multiple tests):</b>', h3_style),
        B('Reparameterisation invariance (Tests A, B): float32 only, single seed, but math is exact'),
        B('SVD singular values predict pruning impact (G, U2, Z): r=0.77-0.86, two seeds, confirmed'),
        B('Composite criterion (Sigma x W2-col) beats SV alone (U2, Z): consistent across two seeds'),
        B('Scaffold neurons need nonzero W2 init for gradient flow (K, P): mechanistically clear'),
        B('LN+tanh scaffold is NOT inert (AJ): max diff 0.086 vs 0.000003 for Iso -- unambiguous'),
        B('IsoGELU fails at lr=0.08; works at lr=0.001 (AK, AQ): direct causal test'),
        Spacer(1, 0.3*cm),
        P('<b>Moderately robust (single seed but large effect or mechanistically explained):</b>', h3_style),
        B('Iso depth stability (E, M, Q): +2.7-4.1%/layer vs Base -2-6%/layer -- large effect'),
        B('Base depth failure is structural, not gradient vanishing (AD, X): mechanistic explanation'),
        B('LN+tanh outperforms Iso at short training; Iso catches up at 100ep (AE, AI): confirmed direction, LR schedule may change magnitude'),
        B('Hybrid Iso-first beats pure options (AL, AR): replicated at two widths, position-dependent'),
        B('Interleaved diagonalise safe; Adam reset harmful (AN): 1L only, mechanism is plausible'),
        Spacer(1, 0.3*cm),
        P('<b>Fragile (single seed, small effect, methodological concerns):</b>', h3_style),
        B('Dynamic topology does not add accuracy (J): single seed, parameter mismatch, short training'),
        B('Overabundance adds +0.14% (W): single/two seeds, inconsistent direction, within noise'),
        B('Pruning timing is best at ep24 (V): single seed, sensitive to training schedule'),
        B('Iso-first-4L at 52.28% is best config (AR): single seed, not confirmed with LR schedule'),
        B('Shell collapse is a genuine gap in Appendix C (AB): finite-sample confound unresolved'),
        Spacer(1, 0.5*cm),
        P(
            '<b>The single most impactful next step</b>: run seeds [42, 123, 7, 0, 99] on '
            'the 10 most important experiments. Every conclusion in this project currently rests '
            'on 1-2 seeds. Five seeds would transform the evidential basis from "plausible" '
            'to "defensible."',
            severe_style
        ),
    ]

    doc.build(story)
    print(f'PDF written to: {OUTPUT}')


if __name__ == '__main__':
    build_pdf()
