from . import KarelForSynthesisParser, KarelWithCurlyParser

PRODS_COMMON = ["prog", "stmt", "stmt_stmt", "action"]
PRODS_CONDITIONAL = ["if", "ifelse", "cond", "cond_without_not"]
PRODS_LOOP = ["repeat", "cste"]  # "while",
ALL_PRODS = PRODS_COMMON + PRODS_CONDITIONAL + PRODS_LOOP


class KarelConfigurableParserMixin:
    def __init__(
        self,
        use_conditionals,
        use_loops,
        use_markers,
        tokens_common,
        tokens_conditional,
        tokens_loop,
        tokens_marker,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.all_tokens = (
            tokens_common + tokens_conditional + tokens_loop + tokens_marker + ALL_PRODS
        )

        tokens = tokens_common + PRODS_COMMON
        if use_conditionals:
            tokens += tokens_conditional
            tokens += PRODS_CONDITIONAL
        if use_loops:
            tokens += tokens_loop
            tokens += PRODS_LOOP
        if use_markers:
            tokens += tokens_marker
        self.syn_tokens = tokens

    def filter_candidates(self, candidates):
        """
        Filters a list of candidates for those with tokens / productions
        in the configuration. Used when generated random programs.
        """
        filtered = []
        for prod in candidates:
            for term in prod.prod:
                # if term not in self.all_tokens:
                #    print(term)
                # assert term in self.all_tokens
                if term not in self.syn_tokens:
                    break
            else:
                filtered.append(prod)
        return filtered


TOKENS_SYNTHESIS_COMMON = [
    "DEF",
    "RUN",
    "M_LBRACE",
    "M_RBRACE",
    "INT",  #'NEWLINE', 'SEMI',
    "MOVE",
    "TURN_RIGHT",
    "TURN_LEFT",
]

TOKENS_SYNTHESIS_CONDITIONAL = [
    "I_LBRACE",
    "I_RBRACE",
    "E_LBRACE",
    "E_RBRACE",
    "IF",
    "IFELSE",
    "ELSE",
    "C_LBRACE",
    "C_RBRACE",
    "FRONT_IS_CLEAR",
    "LEFT_IS_CLEAR",
    "RIGHT_IS_CLEAR",
    "NOT",
    "MARKERS_PRESENT",
    "NO_MARKERS_PRESENT",
]

TOKENS_SYNTHESIS_LOOP = [
    #'W_LBRACE', 'W_RBRACE',
    "R_LBRACE",
    "R_RBRACE",
    #'WHILE',
    "REPEAT",
]

TOKENS_SYNTHESIS_MARKER = [
    "PICK_MARKER",
    "PUT_MARKER",
]


class KarelForSynthesisConfigurableParser(
    KarelConfigurableParserMixin, KarelForSynthesisParser
):
    def __init__(
        self, *args, use_conditionals=True, use_loops=True, use_markers=True, **kwargs
    ):
        super().__init__(
            use_conditionals,
            use_loops,
            use_markers,
            TOKENS_SYNTHESIS_COMMON,
            TOKENS_SYNTHESIS_CONDITIONAL,
            TOKENS_SYNTHESIS_LOOP,
            TOKENS_SYNTHESIS_MARKER,
            *args,
            **kwargs
        )


TOKENS_CURLY_COMMON = [
    "DEF",
    "RUN",
    "LPAREN",
    "RPAREN",
    "LBRACE",
    "RBRACE",
    "SEMI",
    "INT",  #'NEWLINE',
    "MOVE",
    "TURN_RIGHT",
    "TURN_LEFT",
]

TOKENS_CURLY_CONDITIONAL = [
    "IF",
    "IFELSE",
    "ELSE",
    "FRONT_IS_CLEAR",
    "LEFT_IS_CLEAR",
    "RIGHT_IS_CLEAR",
    "NOT",
    "MARKERS_PRESENT",
    "NO_MARKERS_PRESENT",
]

TOKENS_CURLY_LOOP = [
    #'WHILE',
    "REPEAT",
]

TOKENS_CURLY_MARKER = [
    "PICK_MARKER",
    "PUT_MARKER",
]


class KarelWithCurlyConfigurableParser(
    KarelConfigurableParserMixin, KarelWithCurlyParser
):
    def __init__(
        self, *args, use_conditionals=True, use_loops=True, use_markers=True, **kwargs
    ):
        super().__init__(
            use_conditionals,
            use_loops,
            use_markers,
            TOKENS_CURLY_COMMON,
            TOKENS_CURLY_CONDITIONAL,
            TOKENS_CURLY_LOOP,
            TOKENS_CURLY_MARKER,
            *args,
            **kwargs
        )
