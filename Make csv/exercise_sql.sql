create database exercise
;
use exercise
;
create table kospi_ex(
			 trd_date CHAR(10),
             gicode CHAR(7),
             co_nm  VARCHAR(100) charset utf8,
             equity  double DEFAULT NULL,
             ni  double DEFAULT NULL,
             cash_div_com  double DEFAULT NULL,
             liq_equity double DEFAULT NULL,
             liq_debt double DEFAULT NULL,
             inventory double DEFAULT NULL,
             sales double DEFAULT NULL,
             gross_profit double DEFAULT NULL,
             ope_profit double DEFAULT NULL,
             asset double DEFAULT NULL,
             unliq_debt double DEFAULT NULL,
             int_cost double DEFAULT NULL,
             cfo double DEFAULT NULL,
             cgs double DEFAULT NULL,
             receivable double DEFAULT NULL,
             adj_prc double DEFAULT NULL,
             market_cap double DEFAULT NULL,
             cap_size double DEFAULT NULL,
             wics_big double DEFAULT NULL,
             wics_mid double  DEFAULT NULL  
);													
select *
from kospi_ex
;