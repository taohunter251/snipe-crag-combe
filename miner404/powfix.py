"""Parenthesize `-x ** 2` (a SyntaxError in JavaScript) as `-(x ** 2)`, the reading of mathematics and Python.

Token-level: strings, template literals, comments and regular expressions are skipped; for each `**` whose left operand
(a postfix chain: name, number, string, (..), [..], .name, ?.name, call, index) follows a unary operator
(- + ! ~ typeof void delete await), the operand and the whole right side (unary prefixes, a postfix chain, further `**`)
are wrapped in parentheses. Text outside the inserted parentheses is never changed.
"""
from __future__ import annotations

import re

_PUNCT = sorted(""">>>= ... === !== **= <<= >>= >>> ?.( ?.[ => == != <= >= && || ?? ++ -- += -= *= /= %= &= |= ^= << >> ** ?.
{ } ( ) [ ] ; , < > + - * / % & | ^ ! ~ ? : = . @ #""".split(), key=len, reverse=True)
_WORD = re.compile(r"[A-Za-z_$\u0080-￿][\w$\u0080-￿]*")
_NUM = re.compile(r"(?:0[xXoObB][0-9a-fA-F_]+n?|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?n?)")
_UNARY_WORDS = {"typeof", "void", "delete", "await"}
_KEYWORDS_BEFORE_EXPR = {"return", "typeof", "void", "delete", "await", "in", "of", "instanceof", "new", "case", "do",
                         "else", "throw", "yield"}
_OPEN = {"(": ")", "[": "]", "{": "}", "?.(": ")", "?.[": "]"}


def _tokens(src: str) -> list[tuple[str, int, int]] | None:
    """(kind, start, end) for every token; None when the source cannot be scanned (left unchanged)."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(src)
    braces: list[str] = []  # "{" for a block, "`" for a template substitution

    def ends_operand() -> bool:
        if not out:
            return False
        kind, s, e = out[-1]
        text = src[s:e]
        if kind in ("num", "str", "tmpl", "re"):
            return True
        if kind == "word":
            return text not in _KEYWORDS_BEFORE_EXPR
        return text in (")", "]", "}")

    while i < n:
        c = src[i]
        if c in " \t\r\n﻿   ":
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            if j < 0:
                return None
            i = j + 2
            continue
        if c in "'\"":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":
                    return None
                j += 1
            if j >= n:
                return None
            out.append(("str", i, j + 1))
            i = j + 1
            continue
        if c == "`" or (c == "}" and braces and braces[-1] == "`"):
            if c == "}":
                braces.pop()
            j = i + 1
            while j < n and src[j] != "`":
                if src[j] == "\\":
                    j += 1
                elif src.startswith("${", j):
                    break
                j += 1
            if j >= n:
                return None
            if src[j] == "`":
                out.append(("tmpl", i, j + 1))
                i = j + 1
            else:
                out.append(("tmplh", i, j + 2))  # a template part ending in ${: an expression follows
                braces.append("`")
                i = j + 2
            continue
        if c == "/" and not ends_operand():
            j, klass = i + 1, False
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "\n":
                    return None
                if src[j] == "[":
                    klass = True
                elif src[j] == "]":
                    klass = False
                elif src[j] == "/" and not klass:
                    break
                j += 1
            if j >= n:
                return None
            j += 1
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            out.append(("re", i, j))
            i = j
            continue
        m = _NUM.match(src, i) if (c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit())) else None
        if m:
            out.append(("num", i, m.end()))
            i = m.end()
            continue
        m = _WORD.match(src, i)
        if m:
            out.append(("word", i, m.end()))
            i = m.end()
            continue
        for p in _PUNCT:
            if src.startswith(p, i):
                if p == "?." and i + 2 < n and src[i + 2].isdigit():
                    continue  # a ? .5 : b
                if p == "{":
                    braces.append("{")
                elif p == "}":
                    if braces:
                        braces.pop()
                out.append(("p", i, i + len(p)))
                i += len(p)
                break
        else:
            return None
    return out


def _match_back(toks, src, k):
    """Index of the bracket opening the closing bracket at k, or None."""
    close = src[toks[k][1]:toks[k][2]]
    depth = 0
    for j in range(k, -1, -1):
        t = src[toks[j][1]:toks[j][2]]
        if t == close:
            depth += 1
        elif _OPEN.get(t) == close:
            depth -= 1
            if depth == 0:
                return j
    return None


def _match_fwd(toks, src, k):
    close = _OPEN[src[toks[k][1]:toks[k][2]]]
    depth = 0
    for j in range(k, len(toks)):
        t = src[toks[j][1]:toks[j][2]]
        if t in _OPEN and _OPEN[t] == close:
            depth += 1
        elif t == close:
            depth -= 1
            if depth == 0:
                return j
    return None


def _operand_start(toks, src, k):
    """First token of the postfix chain that ends at token k (the left side of `**`), or None."""
    j = k
    while True:
        kind, s, e = toks[j]
        t = src[s:e]
        if t in (")", "]"):
            o = _match_back(toks, src, j)
            if o is None:
                return None
            j = o
            prev = src[toks[j - 1][1]:toks[j - 1][2]] if j > 0 else ""
            pk = toks[j - 1][0] if j > 0 else ""
            # a call or index continues the chain to its callee
            if j > 0 and t in (")", "]") and src[toks[j][1]:toks[j][2]] in ("(", "[", "?.(", "?.[") and (
                    (pk == "word" and prev not in _KEYWORDS_BEFORE_EXPR and prev not in _UNARY_WORDS)
                    or prev in (")", "]") or pk in ("str", "tmpl")):
                j -= 1
                continue
            if src[toks[j][1]:toks[j][2]] in ("?.(", "?.["):
                j -= 1
                continue
            return j
        if kind in ("word", "num", "str", "tmpl"):
            if kind == "word" and (t in _KEYWORDS_BEFORE_EXPR or t in _UNARY_WORDS):
                return None
            if j > 0 and src[toks[j - 1][1]:toks[j - 1][2]] in (".", "?."):
                j -= 2
                if j < 0:
                    return None
                continue
            return j
        return None


def _operand_end(toks, src, k):
    """Last token of the right side of `**` starting at token k: unary prefixes, a postfix chain, further `**`."""
    j = k
    while j < len(toks) and (src[toks[j][1]:toks[j][2]] in ("-", "+", "!", "~") or
                             (toks[j][0] == "word" and src[toks[j][1]:toks[j][2]] in _UNARY_WORDS)):
        j += 1
    if j >= len(toks):
        return None
    kind, s, e = toks[j]
    t = src[s:e]
    if t in ("(", "["):
        j = _match_fwd(toks, src, j)
        if j is None:
            return None
    elif kind == "word" and t == "new":
        return None
    elif kind not in ("word", "num", "str", "tmpl"):
        return None
    while j + 1 < len(toks):
        t = src[toks[j + 1][1]:toks[j + 1][2]]
        if t in (".", "?.") and j + 2 < len(toks) and toks[j + 2][0] == "word":
            j += 2
        elif t in ("(", "[", "?.(", "?.["):
            j = _match_fwd(toks, src, j + 1)
            if j is None:
                return None
        elif t in ("++", "--"):
            j += 1
        else:
            break
    if j + 1 < len(toks) and src[toks[j + 1][1]:toks[j + 1][2]] == "**":
        return _operand_end(toks, src, j + 2)
    return j


def _unary_before(toks, src, j):
    """True when token j - 1 is a unary operator applied to the operand starting at token j."""
    if j == 0:
        return False
    kind, s, e = toks[j - 1]
    t = src[s:e]
    if kind == "word":
        return t in _UNARY_WORDS
    if t not in ("-", "+", "!", "~"):
        return False
    if t in ("!", "~"):
        return True
    if j - 2 < 0:
        return True
    pk, ps, pe = toks[j - 2]
    pt = src[ps:pe]
    if pk in ("num", "str", "tmpl", "re"):
        return False
    if pk == "word":
        return pt in _KEYWORDS_BEFORE_EXPR
    return pt not in (")", "]", "}", "++", "--")


def fix_unary_pow(src: str) -> str:
    """The source with every `<unary> a ** b` written as `<unary> (a ** b)`; unchanged when nothing applies or on doubt."""
    for _ in range(50):
        toks = _tokens(src)
        if toks is None:
            return src
        edit = None
        for k, (kind, s, e) in enumerate(toks):
            if kind != "p" or src[s:e] != "**" or k == 0:
                continue
            start = _operand_start(toks, src, k - 1)
            if start is None or not _unary_before(toks, src, start):
                continue
            end = _operand_end(toks, src, k + 1)
            if end is None:
                continue
            edit = (toks[start][1], toks[end][2])
            break
        if edit is None:
            return src
        a, b = edit
        src = src[:a] + "(" + src[a:b] + ")" + src[b:]
    return src
