<?php

use Espo\Core\Container;
use Espo\Core\InjectableFactory;
use Espo\Core\Utils\Config;
use Espo\Core\Utils\Config\ConfigWriter;

class AfterInstall
{
    public function run(Container $container): void
    {
        $config = $container->getByClass(Config::class);
        $configWriter = $container
            ->getByClass(InjectableFactory::class)
            ->create(ConfigWriter::class);

        $tabList = $config->get('tabList') ?? [];

        foreach (['MvaPolicy', 'MvaClaim'] as $scope) {
            if (!in_array($scope, $tabList, true)) {
                $tabList[] = $scope;
            }
        }

        $configWriter->set('tabList', $tabList);
        $configWriter->save();
    }
}
