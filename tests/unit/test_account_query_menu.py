from core.feishu_bot import messages


def test_aliyun_account_query_menu_opens_ram_query_form():
    assert messages._is_ram_query_entry_intent("阿里云账户查询")
    assert messages._is_ram_query_entry_intent("阿里云账号查询")


def test_volcano_account_query_menu_has_separate_intent():
    assert messages._is_volcano_account_query_entry_intent("火山引擎账户查询")
    assert messages._is_volcano_account_query_entry_intent("火山引擎账号查询")

def test_short_account_query_menu_labels_route():
    assert messages._is_ram_query_entry_intent("阿里云RAM")
    assert messages._is_ram_query_entry_intent("阿里云 RAM")
    assert messages._is_volcano_account_query_entry_intent("火山引擎IAM")
    assert messages._is_volcano_account_query_entry_intent("火山引擎 IAM")
