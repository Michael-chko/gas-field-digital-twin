# Критерии приёмки ТЗ и тесты

Каждый критерий из раздела ТЗ «Порядок работы для Claude Code и критерии приёмки» проверяется
автоматическим тестом. Запуск всех тестов: `pytest`; одного — `pytest "tests/<файл>::<тест>"`.

| № | Критерий ТЗ | Тест | Этап |
| --- | --- | --- | --- |
| 1 | Файл с ошибочной строкой загружается; строка помечена и видна в отчёте о приёме | `tests/test_ingest.py::test_file_with_bad_row_loads_and_row_is_reported`, `::test_generated_samples_all_format_errors_found` | 1 |
| 2 | На синтетике помечены все внедрённые простои, нечисловые и нефизичные значения, аномальные давления при нулевом расходе; выбросы помечены, но не удалены | `tests/test_quality.py::test_all_downtimes_flagged`, `::test_all_non_numeric_flagged`, `::test_all_nonphysical_flagged`, `::test_all_zero_flow_pressures_flagged`, `::test_outliers_flagged_not_removed` | 2 |
| 3 | Пропуск в строке с одним недостающим параметром восстановлен и помечен; строки с несколькими пропусками только помечены | `tests/test_quality.py::test_single_gap_restored_multi_gap_only_flagged`, `::test_restored_value_is_interpolated` | 2 |
| 4 | Отчёт о качестве: доля строк без пропусков, доля пропусков по параметрам, устаревшие потоки | `tests/test_quality.py::test_quality_report_contents` | 2 |
| 5 | П13 — поток не обновлялся дольше двух интервалов; П14 — корреляция связанных датчиков ниже порога | `tests/test_rules.py::test_p13_two_intervals_boundary`, `::test_p14_fires_on_decorrelation`, `::test_p14_threshold_boundary` | 2 |
| 6 | П1: 2 положительных за 20 минут — нет предупреждения; 3 — со звуком; в теневом режиме — журнал без звука | `tests/test_hydrates.py::test_p1_two_positives_no_warning`, `::test_p1_three_positives_warning_with_sound`, `::test_p1_shadow_logged_without_sound` | 4 |
| 7 | М3 обучается на синтетических эпизодах и выводит матрицу ошибок, точность, полноту, F1 и долю правильных ответов | `tests/test_hydrates.py::test_model_trains_and_reports_metrics` | 4 |
| 8 | П2: 0,5, 2/3 и 1,0 от предела — зелёный, жёлтый, красный; 2/3 — жёлтый, предел — красный | `tests/test_geotech_rules.py::test_p2_classes` | 3 |
| 9 | Для П3–П11 и П16 — тест чуть ниже и чуть выше порога | `tests/test_geotech_rules.py::test_p3_rate_threshold`, `::test_threshold_just_below_and_above` (П4, П5, П6, П8, П9, П10), `::test_p7_power_drop`, `::test_p7_efficiency_drop`, `::test_p11_maintenance_overdue`; `tests/test_plant_stops.py::test_p16_below_above` | 3, 8 |
| 10 | М4: деление 60 / 40 %, средняя относительная ошибка, дообучение; после адаптации коэффициенты в прогнозе не меняются | `tests/test_chudin.py::test_split_60_40_and_mean_relative_error`, `::test_retrain_on_new_data`, `::test_adaptation_fixed_during_forecast` | 6 |
| 11 | Заглушку ядра можно заменить другим адаптером без правки кода; рекомендация появляется на экране главного инженера через П12 | `tests/test_cores.py::test_stub_replaceable_without_code_change`, `::test_p12_on_chief_screen_and_decision` | 7 |
| 12 | М10 находит на синтетике все остановки не короче заданной длительности с монотонным ростом давления | `tests/test_plant_stops.py::test_all_long_monotonic_stops_found` | 8 |
| 13 | Причина аномалии, указанная человеком, сохраняется в журнале и выгружается как разметка | `tests/test_journal.py::test_decision_reason_saved_and_exported` | 1 |
| 14 | При проигрывании архива П1 появляется на экране оператора, звучит сигнал, решение в журнале; видны признаки устаревших данных и даты последних замеров | `tests/test_screens_playback.py::test_p1_on_operator_screen_with_sound_and_decision`, `::test_stale_flags_and_last_measurement_dates` | 5 |
| 15 | После установки зависимостей всё работает с отключённой сетью | `tests/test_offline.py::test_full_chain_without_network`, `tests/test_screens_playback.py::test_pages_have_no_internet_dependencies`; установка из `wheels/` с заблокированной сетью проверена вручную | 10 |
| 16 | Пороги и параметры очистки меняются в конфигурации без правки кода | `tests/test_rules.py::test_p13_factor_from_config`, `tests/test_quality.py::test_user_range_narrows_physical`, `::test_smoothing_removes_points_off_polynomial`, `tests/test_hydrates.py::test_p1_threshold_from_config`, `tests/test_geotech_rules.py::test_p10_near_zero_configurable` | 2–4 |
| 17 | У адаптеров шлюза нет методов записи; двойник не может изменить значение в имитаторе OPC UA | `tests/test_gateway.py::test_adapters_have_no_write_methods`, `::test_twin_cannot_change_value_in_simulator` | 9 |
| 18 | Каждый сигнал имитатора привязан в словаре к объекту двойника; сигнал без привязки — в отчёте о приёме | `tests/test_gateway.py::test_every_simulator_signal_bound_and_unbound_reported` | 9 |
| 19 | При обрыве связи данные копятся в буфере и досылаются без потерь и дублей | `tests/test_gateway.py::test_disconnect_backfill_without_loss_or_duplicates`, `::test_central_link_failure_keeps_buffer`, `::test_mqtt_remote_object_to_db_without_duplicates` | 9 |
| 20 | Ручной ввод сохраняется с пометкой «ручной» и проходит те же проверки качества | `tests/test_gateway.py::test_manual_input_marked_and_checked` | 9 |
| 21 | При отказе одного из двух независимых источников приём продолжается со второго | `tests/test_gateway.py::test_failover_to_second_source` | 9 |
| 22 | Переход на другой сервер OPC UA — только адрес и словарь | `tests/test_gateway.py::test_other_server_needs_only_address_and_dictionary` | 9 |
| 23 | Сигнализация АСУ и уведомления двойника хранятся раздельно и помечены источником на экране оператора | `tests/test_gateway.py::test_asu_alarms_separate_on_operator_screen` | 9 |
| 24 | После подтверждения рекомендации двойник по данным АСУ отмечает в журнале, изменился ли режим | `tests/test_gateway.py::test_execution_check_after_confirmation` | 9 |
| 25 | Сигнал без уставок в словаре попадает в отчёт о приёме как незаполненный | `tests/test_gateway.py::test_every_simulator_signal_bound_and_unbound_reported` | 9 |

Сверх ТЗ: адаптер Aspen HYSYS для заводской части — `tests/test_cores.py::test_hysys_*`
(проверка на модели-заменителе; реальная связь — под Windows с лицензией HYSYS, `docs/hysys.md`).
