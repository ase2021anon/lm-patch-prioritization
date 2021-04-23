EOT = '</t>'
TMP_SEP_TOK = '<SEP>'
SEP_TOK_LEN = len(TMP_SEP_TOK)
wrap_sep = lambda x: TMP_SEP_TOK + x + TMP_SEP_TOK
unwrap_sep = lambda x: x[SEP_TOK_LEN:-SEP_TOK_LEN]