"""B1h (#32): nhận diện diff chỉ đổi comment/khoảng trắng. Sửa theo #54: so từng dòng, giữ directive, thụt lề có nghĩa."""
import re
from pathlib import Path

STR = r'"(?:\\.|[^"\\\n])*"|' + r"'(?:\\.|[^'\\\n])*'|`[^`]*`"
C_COM, H_COM = r"//[^\n]*|/\*[\s\S]*?\*/", r"#[^\n]*"
FAMILY = {**{e: C_COM for e in [".java", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp", ".js", ".mjs", ".cjs", ".ts",
                                ".tsx", ".go", ".rs", ".proto"]},
          ".php": C_COM + r"|#(?!\[)[^\n]*",  # #[...] là attribute của PHP 8, không phải comment
          **{e: H_COM for e in [".py", ".rb", ".yaml", ".yml", ".sh"]}}
# Comment mà trình biên dịch/thông dịch đọc: giữ làm token (#54)
DIRECTIVE = re.compile(r"//(?:go:|line |export |\s*\+build|\s*@ts-|# sourceMappingURL=)|///\s*<(?:reference|amd)|"
                       r"/\*\s*webpack|#!|#.*coding[:=]|#\s*(?:frozen_string_literal|warn_indent|shareable_constant_value):")
INDENT = {".py", ".yaml", ".yml"}  # thụt lề đổi nghĩa chương trình


def _line(line, com, indent):
    """Token của một dòng, bỏ comment trừ directive. Dòng chỉ có comment hoặc trắng → []."""
    toks = [t for t in (m.group(0) for m in re.finditer(rf"(?:{STR})|(?:{com})|\w+|\S", line))
            if not re.fullmatch(com, t) or DIRECTIVE.match(t)]
    return [line[:len(line) - len(line.lstrip())], *toks] if toks and indent else toks


def comment_only(changes):
    """#32: diff chỉ đổi comment/khoảng trắng. So từng dòng đổi (không ghép dòng: diff không cho biết các dòng có liền
    nhau không, #54); không chắc (ngôn ngữ lạ, dòng giữa block comment, comment nhiều dòng) thì trả False, tức vẫn chạy e2e."""
    if not changes:
        return False
    for c in changes:
        ext = Path(c.path).suffix
        com = FAMILY.get(ext)
        if com is None:
            return False
        side = [[t for t in (_line(x, com, ext in INDENT) for x in lines) if t] for lines in (c.minus, c.plus)]
        if side[0] != side[1]:
            return False
    return True
